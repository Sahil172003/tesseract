const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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
      const url = `${API_BASE_URL}/api/image-generation`
      console.log('Sending request to:', url)
      console.log('Request params:', params)
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      })

      console.log('Response status:', response.status)
      console.log('Response headers:', Object.fromEntries(response.headers.entries()))
      
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
  },

  // Add a test method
  async testConnection(): Promise<{ message: string }> {
    const url = `${API_BASE_URL}/api/image-generation/test`
    console.log('Testing connection to:', url)
    
    const response = await fetch(url)
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    return response.json()
  }
}