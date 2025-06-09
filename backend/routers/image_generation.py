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
        # Get Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment variables")
        
        # Initialize Gemini client with API key
        client = genai.Client(api_key=api_key)
        
        generated_images = []
        
        # Generate the requested number of images
        for i in range(request.n):
            try:
                print(f"Generating image {i+1} with prompt: {request.prompt}")
                
                # Use the new Gemini 2.0 Flash Preview Image Generation model
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=request.prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE']
                    )
                )
                
                # Extract image data from response
                image_found = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        try:
                            # Decode base64 data
                            image_data = base64.b64decode(part.inline_data.data)
                            
                            # Convert to PIL Image for potential resizing
                            image = Image.open(BytesIO(image_data))
                            
                            # Resize if needed based on request.size
                            if request.size != "512x512":
                                width, height = map(int, request.size.split('x'))
                                image = image.resize((width, height), Image.Resampling.LANCZOS)
                            
                            # Convert back to base64
                            buffer = BytesIO()
                            image.save(buffer, format='PNG')
                            buffer.seek(0)
                            
                            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            data_url = f"data:image/png;base64,{base64_image}"
                            generated_images.append(data_url)
                            image_found = True
                            print(f"Successfully generated image {i+1}")
                            break
                            
                        except Exception as img_error:
                            print(f"Error processing image data for image {i+1}: {str(img_error)}")
                            continue
                
                if not image_found:
                    print(f"No image data found in response for image {i+1}")
                    # Check if there's text response
                    for part in response.candidates[0].content.parts:
                        if part.text is not None:
                            print(f"Text response: {part.text}")
                    raise Exception("No image data found in Gemini response")
                    
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                # For debugging - continue with other images but log the error
                continue
        
        if not generated_images:
            raise HTTPException(status_code=500, detail="Failed to generate any images with Gemini API")
        
        return ImageGenerationResponse(
            images=generated_images,
            prompt=request.prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@router.get("/test")
async def test_endpoint():
    return {"message": "Image generation endpoint is working"}

@router.get("/test-gemini")
async def test_gemini():
    """Test endpoint to verify Gemini image generation is working"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found"}
        
        client = genai.Client(api_key=api_key)
        
        # Simple test generation
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents="A simple red circle on white background",
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # Check response
        has_image = False
        has_text = False
        
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                has_image = True
            if part.text is not None:
                has_text = True
        
        return {
            "status": "success",
            "has_image": has_image,
            "has_text": has_text,
            "api_key_present": bool(api_key)
        }
        
    except Exception as e:
        return {"error": str(e)}