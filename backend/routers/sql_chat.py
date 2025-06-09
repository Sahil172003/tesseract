from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import sqlite3
from io import BytesIO
import json
import uuid
from datetime import datetime
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase

router = APIRouter(prefix="/api/sql-chat", tags=["sql-chat"])

# Store active sessions in memory (in production, use Redis or database)
active_sessions: Dict[str, Dict] = {}

class DatabaseInfo(BaseModel):
    tables: List[str]
    table_info: Dict[str, str]

class QueryRequest(BaseModel):
    question: str
    session_id: str

class QueryResponse(BaseModel):
    response: str
    data: Optional[List[Dict[str, Any]]] = None
    sql_query: Optional[str] = None
    timestamp: str

class DatabaseUploadResponse(BaseModel):
    session_id: str
    database_info: DatabaseInfo
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str
    data: Optional[List[Dict[str, Any]]] = None
    sql_query: Optional[str] = None
    timestamp: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]
    session_id: str

class SQLChatBot:
    def __init__(self, db_path: str):
        """Initialize the SQL chatbot with a database path"""
        self.db = None
        self.sql_chain = None
        self.db_path = db_path
        self.chat_history = []
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and SQL chain"""
        try:
            # Connect to database
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
            
            # Initialize NVIDIA LLM with Llama 405B
            nvidia_api_key = os.getenv("NVIDIA_NIM_API_KEY")
            if not nvidia_api_key:
                raise ValueError("NVIDIA_NIM_API_KEY not found in environment variables")
            
            llm = ChatNVIDIA(
                model="meta/llama-3.1-405b-instruct",
                api_key=nvidia_api_key,
                temperature=0.1,
                max_tokens=1024
            )
            
            self.sql_chain = SQLDatabaseChain.from_llm(
                llm=llm, 
                db=self.db, 
                verbose=True,
                return_intermediate_steps=True
            )
            
        except Exception as e:
            print(f"Error initializing SQLChatBot: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the database with natural language"""
        if not self.sql_chain:
            return {
                "response": "Error: Database not connected",
                "data": None,
                "sql_query": None
            }
        
        try:
            # Execute the query
            result = self.sql_chain.invoke({"query": question})
            
            response_text = result.get("result", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Extract SQL query from intermediate steps
            sql_query = None
            data = None
            
            if intermediate_steps:
                for step in intermediate_steps:
                    if isinstance(step, dict) and "sql_cmd" in step:
                        sql_query = step["sql_cmd"]
                        break
                    elif isinstance(step, tuple) and len(step) > 0:
                        # Sometimes the SQL is in a tuple format
                        if "SELECT" in str(step[0]).upper():
                            sql_query = str(step[0])
                            break
            
            # If we couldn't extract SQL from intermediate steps, try to extract from response
            if not sql_query and response_text:
                sql_query = self._extract_sql_from_text(response_text)
            
            # Try to extract structured data if it's a SELECT query
            if sql_query and sql_query.strip().upper().startswith("SELECT"):
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(sql_query)
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows:
                            data = [dict(zip(columns, row)) for row in rows]
                except Exception as e:
                    print(f"Error extracting data: {str(e)}")
            
            return {
                "response": response_text,
                "data": data,
                "sql_query": sql_query
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return {
                "response": error_msg,
                "data": None,
                "sql_query": None
            }
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL query from text response"""
        import re
        
        # Look for SQL code blocks with ```sql or ```
        sql_block_pattern = r'```(?:sql)?\s*(.*?)```'
        matches = re.findall(sql_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Take the first SQL block found
            sql_query = matches[0].strip()
            return sql_query
        
        # If no code blocks found, look for SELECT statements directly
        select_pattern = r'(SELECT[\s\S]*?;)'
        select_matches = re.findall(select_pattern, text, re.IGNORECASE)
        
        if select_matches:
            return select_matches[0].strip()
        
        # Look for any SQL keywords pattern
        sql_keywords_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)[\s\S]*?)(?:\n\n|\Z)'
        keyword_matches = re.findall(sql_keywords_pattern, text, re.IGNORECASE)
        
        if keyword_matches:
            sql_candidate = keyword_matches[0].strip()
            # Remove any trailing explanatory text
            sql_candidate = re.split(r'\n(?=\w)', sql_candidate)[0]
            return sql_candidate.rstrip(';') + ';'
        
        return None
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        if not self.db:
            return None
        
        try:
            tables = self.db.get_usable_table_names()
            table_info = {}
            
            for table in tables:
                table_info[table] = self.db.get_table_info([table])
            
            return {
                "tables": tables,
                "table_info": table_info
            }
        except Exception as e:
            print(f"Error getting database info: {str(e)}")
            return None

@router.post("/upload-database", response_model=DatabaseUploadResponse)
async def upload_database(file: UploadFile = File(...)):
    """Upload and process a SQLite database file"""
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            raise HTTPException(
                status_code=400, 
                detail="Only SQLite database files (.db, .sqlite, .sqlite3) are allowed"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create temporary file for this session
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"db_{session_id}.db")
        
        with open(temp_file_path, 'wb') as f:
            f.write(file_content)
        
        try:
            # Initialize SQLChatBot
            chatbot = SQLChatBot(temp_file_path)
            
            # Get database info
            db_info = chatbot.get_database_info()
            if not db_info:
                raise HTTPException(status_code=500, detail="Failed to read database schema")
            
            # Store session data
            active_sessions[session_id] = {
                "chatbot": chatbot,
                "temp_file_path": temp_file_path,
                "filename": file.filename,
                "created_at": datetime.now().isoformat(),
                "chat_history": []
            }
            
            return DatabaseUploadResponse(
                session_id=session_id,
                database_info=DatabaseInfo(
                    tables=db_info["tables"],
                    table_info=db_info["table_info"]
                ),
                message="Database uploaded and processed successfully"
            )
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Database processing failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Query the database using natural language"""
    try:
        # Check if session exists
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a database first.")
        
        session = active_sessions[request.session_id]
        chatbot = session["chatbot"]
        
        # Query the database
        result = chatbot.query(request.question)
        
        # Create response with timestamp
        timestamp = datetime.now().isoformat()
        response = QueryResponse(
            response=result["response"],
            data=result["data"],
            sql_query=result["sql_query"],
            timestamp=timestamp
        )
        
        # Add to chat history
        session["chat_history"].extend([
            {
                "role": "user",
                "content": request.question,
                "data": None,
                "sql_query": None,
                "timestamp": timestamp
            },
            {
                "role": "assistant",
                "content": result["response"],
                "data": result["data"],
                "sql_query": result["sql_query"],
                "timestamp": timestamp
            }
        ])
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        messages = [ChatMessage(**msg) for msg in session["chat_history"]]
        
        return ChatHistoryResponse(
            messages=messages,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session and cleanup resources"""
    try:
        if session_id in active_sessions:
            session = active_sessions[session_id]
            
            # Clean up temporary file
            temp_file_path = session.get("temp_file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            # Remove session
            del active_sessions[session_id]
        
        return {"message": "Session cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

@router.get("/sessions")
async def list_active_sessions():
    """List all active sessions (for debugging)"""
    try:
        sessions_info = []
        for session_id, session in active_sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "filename": session.get("filename"),
                "created_at": session.get("created_at"),
                "message_count": len(session.get("chat_history", []))
            })
        
        return {"active_sessions": sessions_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {"message": "SQL chat endpoint is working"}

@router.get("/test-nvidia")
async def test_nvidia():
    """Test NVIDIA API connection"""
    try:
        nvidia_api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if not nvidia_api_key:
            return {"error": "NVIDIA_NIM_API_KEY not found in environment variables"}
        
        # Test creating LLM instance
        llm = ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",
            api_key=nvidia_api_key,
            temperature=0.1,
            max_tokens=100
        )
        
        # Test a simple query
        test_response = llm.invoke("Hello, this is a test message.")
        
        return {
            "status": "success",
            "api_key_present": bool(nvidia_api_key),
            "test_response": str(test_response.content) if hasattr(test_response, 'content') else str(test_response),
            "message": "NVIDIA API connection test successful"
        }
        
    except Exception as e:
        return {"error": f"NVIDIA API test failed: {str(e)}"}
    
@router.get("/debug")
async def debug_routes():
    """Debug endpoint to check if routes are working"""
    return {
        "message": "SQL chat router is working",
        "available_endpoints": [
            "/api/sql-chat/upload-database",
            "/api/sql-chat/query", 
            "/api/sql-chat/test",
            "/api/sql-chat/test-nvidia"
        ]
    }