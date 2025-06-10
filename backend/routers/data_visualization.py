from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64
import io
import json
import requests
import tempfile
from pathlib import Path
import warnings
import traceback
import sys
import subprocess
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

router = APIRouter(prefix="/api/data-visualization", tags=["data-visualization"])

# Pydantic Models
class DataAnalysisRequest(BaseModel):
    visualization_request: str
    chart_type: Optional[str] = "auto"

class ChatRequest(BaseModel):
    question: str

class CodeGenerationRequest(BaseModel):
    task_type: str  # "visualization" or "analysis"
    user_input: str
    dataset_info: Dict[str, Any]
    requirements: List[str] = []

class DataAnalysisResponse(BaseModel):
    status: str
    message: str
    data_preview: Optional[str] = None
    summary_stats: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[str]] = None
    narrative: Optional[str] = None
    chart_data: Optional[Dict[str, Any]] = None
    table_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    status: str
    response: str
    code_executed: Optional[str] = None
    execution_result: Optional[str] = None

class CodeGenerationResponse(BaseModel):
    status: str
    generated_code: str
    execution_result: Optional[str] = None
    artifacts: Optional[List[str]] = None  # For visualization images

class DatasetInfo:
    """Enhanced dataset information class"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_rows = len(df)
        self.total_columns = len(df.columns)
        self.columns = list(df.columns)
        self.dtypes = dict(df.dtypes.astype(str))
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.missing_values = df.isnull().sum().to_dict()
        self.memory_usage = df.memory_usage(deep=True).sum()
        
    def get_sample_data(self, rows: int = 10) -> str:
        """Get formatted sample data for LLM"""
        sample_df = self.df.head(rows)
        return f"""
Dataset Overview:
- Total rows: {self.total_rows:,}
- Total columns: {self.total_columns}
- Memory usage: {self.memory_usage / 1024 / 1024:.2f} MB

Column Information:
{json.dumps(self.dtypes, indent=2)}

Numeric columns: {self.numeric_columns}
Categorical columns: {self.categorical_columns}
DateTime columns: {self.datetime_columns}

Missing values per column:
{json.dumps({k: v for k, v in self.missing_values.items() if v > 0}, indent=2)}

Sample data (first {rows} rows):
{sample_df.to_string(max_cols=20, max_colwidth=50)}

Summary statistics for numeric columns:
{self.df.describe().to_string() if self.numeric_columns else "No numeric columns found"}
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "missing_values": self.missing_values,
            "memory_usage_mb": round(self.memory_usage / 1024 / 1024, 2)
        }

class LLMCodeGenerator:
    """Enhanced LLM code generator with full context"""
    
    def __init__(self, api_token: str = None):
        self.api_token = api_token or os.getenv("NVIDIA_NIM_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        
    def generate_visualization_code(self, dataset_info: DatasetInfo, user_request: str) -> str:
        """Generate complete visualization code with full context"""
        
        context = dataset_info.get_sample_data()
        
        prompt = f"""
You are an expert data visualization programmer. Generate a complete Python script for data visualization.

DATASET CONTEXT:
{context}

USER REQUEST: {user_request if user_request.strip() else "Create comprehensive visualizations that best represent this dataset"}

REQUIREMENTS:
Generate a complete Python script that:
1. Imports required libraries (pandas, numpy, matplotlib, seaborn)
2. Loads the dataset from 'data.csv' in current directory
3. Creates appropriate visualizations based on the data types and user request
4. Saves each plot as 'plot_1.png', 'plot_2.png', etc.
5. Uses defensive programming (check column existence, handle missing values)
6. Creates 2-5 meaningful visualizations
7. Uses proper figure sizing and styling
8. Closes plots properly to free memory

TEMPLATE STRUCTURE:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data.csv')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Data exploration
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Visualization 1: [Description]
if condition_check:
    plt.figure(figsize=(12, 8))
    # plotting code
    plt.title('Title')
    plt.savefig('plot_1.png', dpi=150, bbox_inches='tight')
    plt.close()

# Continue for other visualizations...
```

Generate the COMPLETE script following this structure. Make it specific to the dataset characteristics.
Return ONLY the Python code, no explanations.
"""

        return self._call_llm(prompt, max_tokens=2500, temperature=0.2)
    
    def generate_analysis_code(self, dataset_info: DatasetInfo, question: str) -> str:
        """Generate complete analysis code with full context"""
        
        context = dataset_info.get_sample_data()
        
        prompt = f"""
You are an expert data analyst programmer. Generate a complete Python script for data analysis.

DATASET CONTEXT:
{context}

USER QUESTION: {question}

REQUIREMENTS:
Generate a complete Python script that:
1. Imports required libraries (pandas, numpy)
2. Loads the dataset from 'data.csv' in current directory
3. Performs analysis to answer the user's question
4. Prints results in a clear, formatted way
5. Uses defensive programming (check column existence, handle missing values)
6. Provides meaningful insights

TEMPLATE STRUCTURE:
```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data.csv')

# Analysis code to answer: {question}
print("="*50)
print(f"ANALYSIS: {question}")
print("="*50)

# Your analysis code here
# Use print() statements to show results

print("\\nAnalysis completed successfully.")
```

Generate the COMPLETE script. Make it specific to answer the user's question.
Return ONLY the Python code, no explanations.
"""

        return self._call_llm(prompt, max_tokens=1500, temperature=0.1)
    
    def format_response(self, question: str, code: str, result: str) -> str:
        """Format analysis response using LLM"""
        
        prompt = f"""
Format this data analysis result into a clear, user-friendly response.

USER QUESTION: {question}
CODE EXECUTED: {code}
RAW RESULT: {result}

Provide a clear, concise explanation that:
1. Directly answers the user's question
2. Explains what the numbers/results mean
3. Highlights key insights
4. Uses plain English, not technical jargon
5. Keeps it under 200 words

Return only the formatted response.
"""

        return self._call_llm(prompt, max_tokens=400, temperature=0.7)
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Call LLM API with error handling"""
        try:
            if not self.api_token:
                logger.warning("No API token available, using fallback")
                return self._get_fallback_response(prompt)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }
            
            data = {
                "model": "meta/llama-3.1-405b-instruct",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert Python programmer specializing in data analysis and visualization. Generate complete, executable code with proper error handling."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.base_url,
                headers=headers, 
                data=json.dumps(data), 
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                
                # Clean up code blocks
                if "```python" in content:
                    content = content.split("```python")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Validate Python syntax for code generation
                if "import" in content and "def " not in prompt.lower():
                    try:
                        compile(content, '<string>', 'exec')
                        return content
                    except SyntaxError as e:
                        logger.error(f"Generated code has syntax errors: {e}")
                        return self._get_fallback_response(prompt)
                
                return content
            else:
                logger.error(f"API request failed: {response.status_code}")
                return self._get_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Provide fallback responses when LLM is unavailable"""
        if "visualization" in prompt.lower():
            return self._get_fallback_visualization_code()
        elif "analysis" in prompt.lower():
            return self._get_fallback_analysis_code()
        else:
            return "Error: Unable to generate response. Please try again."
    
    def _get_fallback_visualization_code(self) -> str:
        """Fallback visualization code"""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
plt.style.use('default')
sns.set_palette("husl")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Plot 1: Distribution of first numeric column
if len(numeric_cols) > 0:
    plt.figure(figsize=(12, 8))
    sns.histplot(df[numeric_cols[0]].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {numeric_cols[0]}', fontsize=16)
    plt.xlabel(numeric_cols[0])
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('plot_1.png', dpi=150, bbox_inches='tight')
    plt.close()

# Plot 2: Correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig('plot_2.png', dpi=150, bbox_inches='tight')
    plt.close()

# Plot 3: Categorical distribution
if len(categorical_cols) > 0:
    plt.figure(figsize=(12, 8))
    value_counts = df[categorical_cols[0]].value_counts().head(15)
    value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {categorical_cols[0]}', fontsize=16)
    plt.xlabel(categorical_cols[0])
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_3.png', dpi=150, bbox_inches='tight')
    plt.close()

# Plot 4: Missing values analysis
plt.figure(figsize=(12, 8))
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    missing_data.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Missing Values by Column', fontsize=16)
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45, ha='right')
else:
    plt.text(0.5, 0.5, 'No Missing Values Found!', 
             ha='center', va='center', transform=plt.gca().transAxes, 
             fontsize=20, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.title('Data Quality Check: Missing Values', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_4.png', dpi=150, bbox_inches='tight')
plt.close()
"""
    
    def _get_fallback_analysis_code(self) -> str:
        """Fallback analysis code"""
        return """
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')

print("="*60)
print("DATASET ANALYSIS SUMMARY")
print("="*60)

print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\\nColumn Information:")
for col in df.columns:
    dtype = str(df[col].dtype)
    non_null = df[col].count()
    null_count = df[col].isnull().sum()
    print(f"  {col}: {dtype} ({non_null:,} non-null, {null_count:,} null)")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print(f"\\nNumeric Columns Summary:")
    print(df[numeric_cols].describe())

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\\nCategorical Columns Summary:")
    for col in categorical_cols[:3]:  # Show first 3 categorical columns
        print(f"\\n{col} - Top 5 values:")
        print(df[col].value_counts().head())

print("\\nAnalysis completed successfully.")
"""

class CodeExecutor:
    """Enhanced code executor with proper isolation"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, 'data.csv')
        self._prepare_data_file()
    
    def _prepare_data_file(self):
        """Copy uploaded file to standard location for code execution"""
        try:
            if self.file_path.endswith('.csv'):
                shutil.copy2(self.file_path, self.data_file)
            else:
                # Convert Excel to CSV for standardization
                df = pd.read_excel(self.file_path)
                df.to_csv(self.data_file, index=False)
        except Exception as e:
            logger.error(f"Error preparing data file: {e}")
            raise
    
    def execute_visualization_code(self, code: str) -> List[str]:
        """Execute visualization code and return base64 encoded images"""
        visualizations = []
        script_path = os.path.join(self.temp_dir, 'viz_script.py')
        
        try:
            # Write code to script file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            logger.info(f"Visualization script output: {result.stdout}")
            if result.stderr:
                logger.warning(f"Visualization script stderr: {result.stderr}")
            
            # Collect generated plots
            plot_files = sorted([
                f for f in os.listdir(self.temp_dir) 
                if f.startswith('plot_') and f.endswith('.png')
            ])
            
            for plot_file in plot_files:
                plot_path = os.path.join(self.temp_dir, plot_file)
                if os.path.exists(plot_path) and os.path.getsize(plot_path) > 0:
                    with open(plot_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        visualizations.append(f"data:image/png;base64,{img_data}")
            
            return visualizations
            
        except subprocess.TimeoutExpired:
            logger.error("Visualization code execution timed out")
            raise Exception("Code execution timed out")
        except Exception as e:
            logger.error(f"Error executing visualization code: {e}")
            raise
    
    def execute_analysis_code(self, code: str) -> str:
        """Execute analysis code and return output"""
        script_path = os.path.join(self.temp_dir, 'analysis_script.py')
        
        try:
            # Write code to script file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip() if result.stdout.strip() else "Code executed successfully (no output)"
            else:
                error_msg = result.stderr.strip() if result.stderr.strip() else "Unknown error"
                logger.error(f"Analysis code execution failed: {error_msg}")
                return f"Error executing code: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error("Analysis code execution timed out")
            return "Error: Code execution timed out"
        except Exception as e:
            logger.error(f"Error executing analysis code: {e}")
            return f"Error executing code: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

class DataAnalyzer:
    """Main data analyzer class with enhanced capabilities"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.dataset_info = None
        self.llm_generator = LLMCodeGenerator()
        self.code_executor = None
        
    def load_data(self) -> bool:
        """Load and analyze the uploaded data file"""
        try:
            file_extension = Path(self.file_path).suffix.lower()
            
            if file_extension == '.csv':
                # Try multiple encodings for CSV files
                encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(self.file_path, encoding=encoding)
                        logger.info(f"CSV loaded successfully with {encoding} encoding")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    raise Exception("Could not read CSV with any supported encoding")
                    
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
                logger.info("Excel file loaded successfully")
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Create dataset info
            self.dataset_info = DatasetInfo(self.df)
            
            # Initialize code executor
            self.code_executor = CodeExecutor(self.file_path)
            
            logger.info(f"Data loaded: {self.dataset_info.total_rows:,} rows, {self.dataset_info.total_columns} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_table_preview(self, rows: int = 20) -> Dict[str, Any]:
        """Get table data for frontend display"""
        if self.df is None or self.dataset_info is None:
            return None
        
        preview_df = self.df.head(rows)
        
        return {
            "columns": self.dataset_info.columns,
            "data": preview_df.to_dict('records'),
            "total_rows": self.dataset_info.total_rows,
            "total_columns": self.dataset_info.total_columns,
            "dataset_info": self.dataset_info.to_dict()
        }
    
    def create_visualizations(self, user_request: str = "") -> List[str]:
        """Generate and execute visualization code"""
        if not self.dataset_info or not self.code_executor:
            raise Exception("No data loaded")
        
        try:
            # Generate visualization code using LLM
            viz_code = self.llm_generator.generate_visualization_code(
                self.dataset_info, 
                user_request
            )
            
            logger.info("Visualization code generated, executing...")
            
            # Execute the code and get visualizations
            visualizations = self.code_executor.execute_visualization_code(viz_code)
            
            if not visualizations:
                logger.warning("No visualizations generated, trying fallback")
                fallback_code = self.llm_generator._get_fallback_visualization_code()
                visualizations = self.code_executor.execute_visualization_code(fallback_code)
            
            logger.info(f"Generated {len(visualizations)} visualizations")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
    
    def answer_question(self, question: str) -> tuple[str, str, str]:
        """Generate and execute analysis code to answer questions"""
        if not self.dataset_info or not self.code_executor:
            raise Exception("No data loaded")
        
        try:
            # Generate analysis code using LLM
            analysis_code = self.llm_generator.generate_analysis_code(
                self.dataset_info, 
                question
            )
            
            logger.info("Analysis code generated, executing...")
            
            # Execute the code and get results
            execution_result = self.code_executor.execute_analysis_code(analysis_code)
            
            # Format the response using LLM
            formatted_response = self.llm_generator.format_response(
                question, 
                analysis_code, 
                execution_result
            )
            
            return formatted_response, analysis_code, execution_result
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def cleanup(self):
        """Clean up all resources"""
        try:
            if self.code_executor:
                self.code_executor.cleanup()
            
            # Clean up the uploaded file
            if self.file_path and os.path.exists(self.file_path):
                file_dir = os.path.dirname(self.file_path)
                shutil.rmtree(file_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Global analyzer instance
current_analyzer: Optional[DataAnalyzer] = None

# API Endpoints
@router.post("/upload", response_model=DataAnalysisResponse)
async def upload_and_analyze_data(file: UploadFile = File(...)):
    """Upload and analyze data file with enhanced processing"""
    global current_analyzer
    
    try:
        # Validate file format
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload CSV or Excel files."
            )
        
        # Create persistent temporary file
        temp_dir = tempfile.mkdtemp()
        file_extension = Path(file.filename).suffix
        temp_file_path = os.path.join(temp_dir, f"uploaded_data{file_extension}")
        
        # Save uploaded file
        content = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
        
        # Clean up previous analyzer
        if current_analyzer:
            current_analyzer.cleanup()
        
        # Create new analyzer and load data
        current_analyzer = DataAnalyzer(temp_file_path)
        
        if not current_analyzer.load_data():
            raise HTTPException(status_code=400, detail="Failed to load data file. Please check file format and content.")
        
        # Get table preview
        table_data = current_analyzer.get_table_preview()
        
        return DataAnalysisResponse(
            status="success",
            message=f"Data uploaded successfully. Loaded {table_data['total_rows']:,} rows and {table_data['total_columns']} columns.",
            table_data=table_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/create-visualization", response_model=DataAnalysisResponse)
async def create_visualization(request: DataAnalysisRequest):
    """Create visualizations using enhanced LLM code generation"""
    global current_analyzer
    
    if current_analyzer is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    try:
        logger.info(f"Creating visualizations with request: {request.visualization_request}")
        
        visualizations = current_analyzer.create_visualizations(request.visualization_request)
        
        if not visualizations:
            raise Exception("No visualizations were generated. Please try a different request or check your data.")
        
        return DataAnalysisResponse(
            status="success",
            message=f"Successfully created {len(visualizations)} visualizations",
            visualizations=visualizations
        )
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    """Chat about the dataset using enhanced LLM analysis"""
    global current_analyzer
    
    if current_analyzer is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    try:
        logger.info(f"Processing chat question: {request.question}")
        
        formatted_response, code_executed, execution_result = current_analyzer.answer_question(request.question)
        
        return ChatResponse(
            status="success",
            response=formatted_response,
            code_executed=code_executed,
            execution_result=execution_result
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            status="error",
            response=f"Sorry, I encountered an error while analyzing your question: {str(e)}. Please try rephrasing your question or check if your data is properly loaded."
        )

@router.post("/generate-code", response_model=CodeGenerationResponse)
async def generate_code_endpoint(request: CodeGenerationRequest):
    """Advanced endpoint for custom code generation"""
    global current_analyzer
    
    if current_analyzer is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    try:
        llm_generator = LLMCodeGenerator()
        
        if request.task_type == "visualization":
            code = llm_generator.generate_visualization_code(
                current_analyzer.dataset_info, 
                request.user_input
            )
            artifacts = current_analyzer.code_executor.execute_visualization_code(code)
            
            return CodeGenerationResponse(
                status="success",
                generated_code=code,
                artifacts=artifacts
            )
            
        elif request.task_type == "analysis":
            code = llm_generator.generate_analysis_code(
                current_analyzer.dataset_info, 
                request.user_input
            )
            result = current_analyzer.code_executor.execute_analysis_code(code)
            
            return CodeGenerationResponse(
                status="success",
                generated_code=code,
                execution_result=result
            )
        else:
            raise ValueError("Invalid task_type. Use 'visualization' or 'analysis'")
            
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@router.get("/dataset-info")
async def get_dataset_info():
    """Get detailed information about the current dataset"""
    global current_analyzer
    
    if current_analyzer is None or current_analyzer.dataset_info is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    return {
        "status": "success",
        "dataset_info": current_analyzer.dataset_info.to_dict(),
        "sample_data": current_analyzer.dataset_info.get_sample_data(5)
    }

@router.post("/clear")
async def clear_data():
    """Clear current data session and cleanup all resources"""
    global current_analyzer
    
    try:
        if current_analyzer:
            current_analyzer.cleanup()
        current_analyzer = None
        
        logger.info("Data session cleared successfully")
        return {"status": "success", "message": "Data cleared successfully"}
        
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
        return {"status": "success", "message": "Data cleared (with warnings)"}

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API functionality"""
    return {
        "status": "success",
        "message": "Data visualization API is running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/upload - Upload data file",
            "/create-visualization - Generate visualizations", 
            "/chat - Ask questions about data",
            "/generate-code - Advanced code generation",
            "/dataset-info - Get dataset information",
            "/clear - Clear current session"
        ]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "current_session": current_analyzer is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }