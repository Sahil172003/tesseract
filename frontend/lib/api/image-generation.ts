const API_BASE_URL = 'http://localhost:8000'

export interface ImageGenerationParams {
  prompt: string
  n?: number
  size?: string
}

export interface ImageGenerationResponse {
  images: string[]
  prompt: string
}

export const imageGenerationApi = {
  async generateImages(params: ImageGenerationParams): Promise<ImageGenerationResponse> {
    const response = await fetch(`${API_BASE_URL}/api/image-generation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || 'Failed to generate images')
    }

    return response.json()
  }
}