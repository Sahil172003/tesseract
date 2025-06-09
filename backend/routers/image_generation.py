from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os
from typing import List, Optional

router = APIRouter(prefix="/api/image-generation", tags=["image-generation"])

# Initialize Gemini client
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "512x512"

class ImageGenerationResponse(BaseModel):
    images: List[str]
    prompt: str

@router.post("", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    try:
        client = get_gemini_client()
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=request.prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        generated_images = []
        
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Convert image data to base64 string
                image_data = part.inline_data.data
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Format as data URL for frontend
                data_url = f"data:image/png;base64,{base64_image}"
                generated_images.append(data_url)
        
        # If we need multiple images, make additional requests
        while len(generated_images) < request.n:
            additional_response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            for part in additional_response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    data_url = f"data:image/png;base64,{base64_image}"
                    generated_images.append(data_url)
                    
                    if len(generated_images) >= request.n:
                        break
        
        return ImageGenerationResponse(
            images=generated_images[:request.n],
            prompt=request.prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@router.get("/test")
async def test_endpoint():
    return {"message": "Image generation endpoint is working"}