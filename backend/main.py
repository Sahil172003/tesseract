from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import routers directly from the routers directory
from routers import image_generation, sql_chat, data_visualization, open_source_summarizer, claude_summarizer, openai_summarizer, gemini_summarizer

app = FastAPI(
    title="Tesseract API",
    description="Backend API for Tesseract application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(image_generation.router)
app.include_router(sql_chat.router)
app.include_router(data_visualization.router)
app.include_router(open_source_summarizer.router)
app.include_router(claude_summarizer.router)
app.include_router(openai_summarizer.router)
app.include_router(gemini_summarizer.router)

@app.get("/")
async def root():
    return {"message": "Tesseract API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/test")
async def api_test():
    return {"message": "API is working", "status": "ok"}

# Add a debug endpoint to list all routes
@app.get("/debug/routes")
async def debug_routes():
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else []
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    print("Starting Tesseract API server...")
    print("Available endpoints:")
    print("- http://localhost:8000/")
    print("- http://localhost:8000/health")
    print("- http://localhost:8000/api/image-generation/test")
    print("- http://localhost:8000/api/data-visualization/test")
    print("- http://localhost:8000/api/document-chat/test")
    print("- http://localhost:8000/api/claude-chat/test")
    print("- http://localhost:8000/api/openai-chat/test")
    print("- http://localhost:8000/api/gemini-chat/test")
    print("- http://localhost:8000/debug/routes")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)