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
    try {
      console.log('Sending request to:', `${API_BASE_URL}/api/image-generation`)
      console.log('Request params:', params)
      
      const response = await fetch(`${API_BASE_URL}/api/image-generation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      })

      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        
        let errorData
        try {
          errorData = JSON.parse(errorText)
        } catch {
          errorData = { detail: errorText || 'Failed to generate images' }
        }
        
        throw new Error(errorData.detail || `HTTP ${response.status}: Failed to generate images`)
      }

      const data = await response.json()
      console.log('Success response:', data)
      return data
      
    } catch (error) {
      console.error('API call failed:', error)
      throw error
    }
  }
}