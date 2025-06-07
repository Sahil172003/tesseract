import { apiClient } from './client';

export interface ImageGenerationRequest {
  prompt: string;
  n?: number;
  size?: string;
}

export interface ImageGenerationResponse {
  images: string[];
  prompt: string;
}

// Image Generation API functions
export const imageGenerationApi = {
  // Generate images
  async generateImages(request: ImageGenerationRequest): Promise<ImageGenerationResponse> {
    return apiClient.post('/image-generation', request);
  }
};
