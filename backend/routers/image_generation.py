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
                
                # Use the correct model for image generation
                # Try different models in order of preference
                models_to_try = [
                    "gemini-2.0-flash-exp",
                    "gemini-exp-1206", 
                    "gemini-2.0-flash-thinking-exp-01-21"
                ]
                
                response = None
                model_used = None
                
                for model in models_to_try:
                    try:
                        print(f"Trying model: {model}")
                        response = client.models.generate_content(
                            model=model,
                            contents=f"Generate an image: {request.prompt}",
                            config=types.GenerateContentConfig(
                                response_modalities=['IMAGE']
                            )
                        )
                        model_used = model
                        print(f"Successfully used model: {model}")
                        break
                    except Exception as model_error:
                        print(f"Model {model} failed: {str(model_error)}")
                        continue
                
                if response is None:
                    raise Exception("All image generation models failed")
                
                print(f"Response received for image {i+1} using {model_used}")
                print(f"Response type: {type(response)}")
                print(f"Response candidates: {len(response.candidates) if response.candidates else 0}")
                
                # Extract image data from response
                image_found = False
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    print(f"Candidate content parts: {len(candidate.content.parts) if candidate.content and candidate.content.parts else 0}")
                    
                    for part_idx, part in enumerate(candidate.content.parts):
                        print(f"Part {part_idx}: type={type(part)}")
                        
                        # Check for inline_data attribute
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            print(f"Found inline_data in part {part_idx}")
                            try:
                                # Get the base64 data
                                image_data = part.inline_data.data
                                mime_type = part.inline_data.mime_type
                                
                                print(f"Image data length: {len(image_data) if image_data else 0}")
                                print(f"MIME type: {mime_type}")
                                
                                # Handle the image data
                                if isinstance(image_data, str):
                                    # Data is already base64 encoded
                                    try:
                                        # Verify it's valid base64
                                        decoded_data = base64.b64decode(image_data)
                                        image = Image.open(BytesIO(decoded_data))
                                        base64_data = image_data
                                    except Exception as decode_error:
                                        print(f"Base64 decode error: {decode_error}")
                                        continue
                                else:
                                    # Data is bytes, encode to base64
                                    try:
                                        image = Image.open(BytesIO(image_data))
                                        base64_data = base64.b64encode(image_data).decode('utf-8')
                                    except Exception as bytes_error:
                                        print(f"Bytes processing error: {bytes_error}")
                                        continue
                                
                                # Resize if needed based on request.size
                                if request.size != "512x512":
                                    try:
                                        width, height = map(int, request.size.split('x'))
                                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                                        
                                        # Re-encode after resize
                                        buffer = BytesIO()
                                        image.save(buffer, format='PNG')
                                        buffer.seek(0)
                                        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                    except Exception as resize_error:
                                        print(f"Resize error: {resize_error}")
                                        # Use original data if resize fails
                                
                                # Create data URL
                                data_url = f"data:image/png;base64,{base64_data}"
                                generated_images.append(data_url)
                                image_found = True
                                print(f"Successfully processed image {i+1}")
                                break
                                
                            except Exception as img_error:
                                print(f"Error processing image data for image {i+1}: {str(img_error)}")
                                import traceback
                                traceback.print_exc()
                                continue
                        else:
                            print(f"Part {part_idx} has no inline_data, attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                
                if not image_found:
                    print(f"No image data found in response for image {i+1}")
                    # Print response structure for debugging
                    if response.candidates and response.candidates[0].content:
                        for idx, part in enumerate(response.candidates[0].content.parts):
                            print(f"Part {idx} attributes: {[attr for attr in dir(part) if not attr.startswith('_')]}")
                    
                    raise Exception(f"No image data found in response from model {model_used}")
                    
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with other images but log the error
                continue
        
        if not generated_images:
            raise HTTPException(status_code=500, detail="Failed to generate any images with Gemini API. Check server logs for details.")
        
        return ImageGenerationResponse(
            images=generated_images,
            prompt=request.prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@router.get("/test")
async def test_endpoint():
    return {"message": "Image generation endpoint is working"}

@router.get("/models")
async def list_available_models():
    """List all available models to see which ones support image generation"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found"}
        
        client = genai.Client(api_key=api_key)
        
        # List all available models
        models = client.models.list()
        
        model_info = []
        for model in models:
            model_data = {
                "name": model.name,
                "display_name": getattr(model, 'display_name', 'N/A'),
                "description": getattr(model, 'description', 'N/A'),
                "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
            }
            model_info.append(model_data)
        
        return {
            "available_models": model_info,
            "total_count": len(model_info)
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@router.get("/test-gemini")
async def test_gemini():
    """Test endpoint to verify Gemini image generation is working"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found"}
        
        client = genai.Client(api_key=api_key)
        
        # Test with different models
        models_to_test = [
            "gemini-2.0-flash-exp",
            "gemini-exp-1206",
            "gemini-2.0-flash-thinking-exp-01-21"
        ]
        
        test_results = {}
        
        for model in models_to_test:
            try:
                print(f"Testing model: {model}")
                response = client.models.generate_content(
                    model=model,
                    contents="Generate an image: A simple red circle on white background",
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE']
                    )
                )
                
                # Check response structure
                has_image = False
                has_text = False
                response_info = {}
                
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for idx, part in enumerate(candidate.content.parts):
                            part_info = {
                                'has_inline_data': hasattr(part, 'inline_data') and part.inline_data is not None,
                                'has_text': hasattr(part, 'text') and part.text is not None,
                                'attributes': [attr for attr in dir(part) if not attr.startswith('_')]
                            }
                            response_info[f'part_{idx}'] = part_info
                            
                            if part_info['has_inline_data']:
                                has_image = True
                            if part_info['has_text']:
                                has_text = True
                
                test_results[model] = {
                    "status": "success",
                    "has_image": has_image,
                    "has_text": has_text,
                    "response_structure": response_info
                }
                
            except Exception as model_error:
                test_results[model] = {
                    "status": "failed",
                    "error": str(model_error)
                }
        
        return {
            "api_key_present": bool(api_key),
            "test_results": test_results
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }