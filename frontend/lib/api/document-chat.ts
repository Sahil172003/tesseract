import { apiClient } from './client';

export interface DocumentQueryRequest {
  text: string;
  document_id?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  message: string;
}

export interface QueryResponse {
  response: string;
}

export interface SQLQueryRequest {
  query: string;
  document_id?: string;
}

export interface VisualizationRequest {
  document_id: string;
  visualization_type: string;
  columns: string[];
}

export interface VisualizationResponse {
  image_url: string;
}

// Document Chat API functions
export const documentChatApi = {
  // Upload a document
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    return apiClient.upload('/document-chat/upload', formData);
  },
  
  // Query a document
  async queryDocument(query: DocumentQueryRequest): Promise<QueryResponse> {
    return apiClient.post('/document-chat/query', query);
  },
  
  // Execute an SQL query
  async executeSqlQuery(query: SQLQueryRequest): Promise<any> {
    return apiClient.post('/document-chat/sqlite/query', query);
  },
  
  // Generate a visualization
  async generateVisualization(request: VisualizationRequest): Promise<VisualizationResponse> {
    return apiClient.post('/document-chat/visualization', request);
  }
};
