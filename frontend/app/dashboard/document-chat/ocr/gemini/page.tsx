"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { FileText, Upload, Send, X, Loader2, FileUp, ChevronDown, Bot, User, RefreshCw, Replace, File, Sparkles } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Progress } from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: string
}

interface DocumentSession {
  session_id: string
  filename: string
  file_format: string
  content_preview: string
  content_length: number
}

interface GeminiModel {
  id: string
  name: string
}

interface DocumentInfo {
  filename: string
  format: string
}

export default function GeminiOCRChatPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [session, setSession] = useState<DocumentSession | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [selectedModel, setSelectedModel] = useState("gemini-1.5-pro")
  const [availableModels, setAvailableModels] = useState<GeminiModel[]>([])
  const [showModelSelector, setShowModelSelector] = useState(true)
  const [currentDocumentInfo, setCurrentDocumentInfo] = useState<DocumentInfo | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels()
  }, [])

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadAvailableModels = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/gemini-chat/models")
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data.models)
      }
    } catch (error) {
      console.error("Failed to load models:", error)
      // Fallback models
      setAvailableModels([
        { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro" },
        { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash" },
        { id: "gemini-1.0-pro", name: "Gemini 1.0 Pro" },
        { id: "gemini-pro", name: "Gemini Pro" },
      ])
    }
  }

  // Handle file upload with better progress handling
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setIsProcessing(true)
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      // Better progress simulation
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 85) {
            clearInterval(progressInterval)
            return 85 // Stop at 85% and let the actual response complete it
          }
          return prev + 15
        })
      }, 300)

      const response = await fetch("http://localhost:8000/api/gemini-chat/upload", {
        method: "POST",
        body: formData,
      })

      // Clear the interval and complete progress
      clearInterval(progressInterval)
      
      if (response.ok) {
        // Animate to 100% quickly
        setUploadProgress(100)
        
        const data = await response.json()
        setSession(data)
        setCurrentDocumentInfo({
          filename: data.filename,
          format: data.file_format
        })
        
        // Small delay to show 100% completion
        setTimeout(() => {
          setIsProcessing(false)
          setUploadProgress(0)
        }, 500)

        toast({
          title: "Document processed successfully",
          description: `You can now chat with ${data.filename} using Gemini AI.`,
          variant: "default",
        })
      } else {
        const error = await response.json()
        throw new Error(error.detail || "Upload failed")
      }
    } catch (error) {
      console.error("Upload error:", error)
      setIsProcessing(false)
      setFile(null)
      setUploadProgress(0)
      
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "An error occurred during upload",
        variant: "destructive",
      })
    }
  }

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    const userMessage: ChatMessage = {
      role: "user",
      content: inputMessage,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputMessage("")
    setIsLoading(true)

    try {
      const response = await fetch("http://localhost:8000/api/gemini-chat/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userMessage.content,
          session_id: session?.session_id || null,
          model: selectedModel,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        const aiMessage: ChatMessage = {
          role: "assistant",
          content: data.message,
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, aiMessage])
        
        // Update document info if provided in response
        if (data.document_info) {
          setCurrentDocumentInfo(data.document_info)
        }
      } else {
        const error = await response.json()
        throw new Error(error.detail || "Chat request failed")
      }
    } catch (error) {
      console.error("Chat error:", error)
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error while processing your question. Please try again.",
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMessage])
      
      toast({
        title: "Chat error",
        description: error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Replace the current document
  const handleReplaceDocument = () => {
    setFile(null)
    setSession(null)
    setCurrentDocumentInfo(null)
    setUploadProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ""
    
    // Trigger file upload immediately
    triggerFileUpload()
  }

  // Clear the current document but keep chat history
  const handleRemoveDocument = () => {
    setFile(null)
    setSession(null)
    setCurrentDocumentInfo(null)
    setUploadProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ""
    
    toast({
      title: "Document removed",
      description: "Document has been removed. You can continue with general chat or upload a new document.",
      variant: "default",
    })
  }

  // Clear all messages
  const handleClearChat = async () => {
    setMessages([])
    
    // Clear general chat history on backend if no document session
    if (!session) {
      try {
        await fetch("http://localhost:8000/api/gemini-chat/chat/clear", {
          method: "DELETE",
        })
      } catch (error) {
        console.error("Failed to clear general chat:", error)
      }
    }
    
    toast({
      title: "Chat cleared",
      description: "All chat messages have been cleared.",
      variant: "default",
    })
  }

  // Trigger file input click
  const triggerFileUpload = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="new">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <h1 className="text-3xl font-bold tracking-tight gradient-text">Gemini AI Assistant</h1>
        <p className="text-muted-foreground mt-1">Chat with Google Gemini AI or upload a document for intelligent document analysis and Q&A.</p>
      </motion.div>

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <motion.div
          className="space-y-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          {/* Model Selector Card - Always visible */}
          <Card className="overflow-hidden">
            <CardContent className="p-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-blue-500" />
                    Gemini Model Selection
                  </h3>
                  <button
                    onClick={() => setShowModelSelector(!showModelSelector)}
                    className="text-xs text-muted-foreground hover:text-primary transition-colors"
                  >
                    <ChevronDown className={`h-4 w-4 transition-transform ${showModelSelector ? 'rotate-180' : ''}`} />
                  </button>
                </div>
                
                <AnimatePresence>
                  {showModelSelector && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select Gemini model" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableModels.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex items-center gap-2">
                                <Sparkles className="h-3 w-3 text-blue-500" />
                                {model.name}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground mt-2">
                        Choose the Gemini model for your conversations. Works for both general chat and document analysis.
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </CardContent>
          </Card>

          {/* Document Upload Card */}
          <Card className="overflow-hidden">
            <CardContent className="p-4">
              {!file ? (
                <motion.div
                  className="flex flex-col items-center justify-center space-y-4 rounded-md border border-dashed p-8"
                  whileHover={{ scale: 1.02 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                >
                  <div className="feature-icon">
                    <FileText className="h-6 w-6" />
                  </div>
                  <div className="space-y-1 text-center">
                    <p className="text-sm font-medium">Upload a document (Optional)</p>
                    <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, CSV, Excel, or image files</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".pdf,.docx,.doc,.txt,.csv,.xlsx,.xls,.png,.jpg,.jpeg,.json,.html,.md,.xml,.pptx,.ppt"
                    onChange={handleFileUpload}
                  />
                  <Button
                    onClick={triggerFileUpload}
                    className="gap-2 shadow-md hover:shadow-lg transition-all duration-300"
                    disabled={isProcessing}
                  >
                    <Upload className="h-4 w-4" />
                    Upload Document
                  </Button>
                </motion.div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                        <FileText className="h-4 w-4 text-primary" />
                      </div>
                      <div>
                        <span className="text-sm font-medium">{file.name}</span>
                        {session && (
                          <p className="text-xs text-muted-foreground">
                            {session.file_format.toUpperCase()} • {Math.round(session.content_length / 1024)}KB
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleReplaceDocument}
                        title="Replace document"
                        className="rounded-full hover:bg-blue-50 hover:text-blue-600"
                        disabled={isProcessing}
                      >
                        <Replace className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={handleRemoveDocument}
                        title="Remove document"
                        className="rounded-full hover:bg-destructive/10 hover:text-destructive"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>

                  {isProcessing ? (
                    <div className="space-y-3 p-4">
                      <div className="flex items-center justify-between text-sm">
                        <span>Processing document...</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} className="h-2" />
                      <div className="flex items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                      </div>
                      <p className="text-xs text-muted-foreground text-center">
                        Analyzing document content and preparing for Gemini AI chat...
                      </p>
                    </div>
                  ) : (
                    <motion.div
                      className="space-y-3"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <div className="rounded-md border p-3">
                        <h3 className="mb-2 text-sm font-medium">Content Preview:</h3>
                        <div className="max-h-[200px] overflow-y-auto text-sm rounded bg-muted/50 p-3">
                          {session?.content_preview || "Processing..."}
                        </div>
                      </div>
                      
                      <div className="text-center">
                        <p className="text-xs text-green-600 font-medium flex items-center justify-center gap-1">
                          <Sparkles className="h-3 w-3" />
                          Document ready for Gemini AI analysis
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          Using {availableModels.find(m => m.id === selectedModel)?.name || selectedModel}
                        </p>
                      </div>
                    </motion.div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          className="flex h-[600px] flex-col rounded-xl border overflow-hidden shadow-md"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {/* Chat Header */}
          <div className="flex items-center justify-between p-4 border-b bg-background/50 backdrop-blur-sm">
            <div className="flex items-center space-x-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
              <div>
                <h3 className="text-sm font-medium">Gemini AI</h3>
                <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                  {currentDocumentInfo ? (
                    <>
                      <File className="h-3 w-3" />
                      <span>Document: {currentDocumentInfo.filename}</span>
                    </>
                  ) : (
                    <span>General conversation mode</span>
                  )}
                </div>
              </div>
            </div>
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClearChat}
                className="text-xs"
              >
                Clear Chat
              </Button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <div className="feature-icon mb-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-r from-blue-500 to-purple-600">
                    <Sparkles className="h-6 w-6 text-white" />
                  </div>
                </div>
                <h3 className="text-lg font-medium">
                  {currentDocumentInfo ? "Chat with your document using Gemini" : "Start a conversation with Gemini"}
                </h3>
                <p className="mt-2 text-sm text-muted-foreground max-w-md">
                  {currentDocumentInfo 
                    ? `Ask questions about ${currentDocumentInfo.filename}. Gemini will analyze the document and provide detailed, accurate answers.`
                    : "Ask me anything! You can have a general conversation with Gemini AI or upload a document for document-specific analysis."
                  }
                </p>
                {!currentDocumentInfo && (
                  <Button
                    onClick={triggerFileUpload}
                    variant="outline"
                    className="mt-4 gap-2"
                  >
                    <FileUp className="h-4 w-4" />
                    Upload Document for Analysis
                  </Button>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <AnimatePresence>
                  {messages.map((message, index) => (
                    <motion.div
                      key={index}
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className={`max-w-[80%] ${message.role === "user" ? "chat-bubble-user" : "chat-bubble-ai"}`}>
                        <div className="flex items-start space-x-2">
                          <div className={`flex-shrink-0 ${message.role === "user" ? "order-2" : ""}`}>
                            {message.role === "user" ? (
                              <User className="h-4 w-4 mt-1" />
                            ) : (
                              <Sparkles className="h-4 w-4 mt-1 text-blue-500" />
                            )}
                          </div>
                          <div className={`flex-1 ${message.role === "user" ? "order-1" : ""}`}>
                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                            <p className="text-xs opacity-60 mt-1">
                              {new Date(message.timestamp).toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {isLoading && (
                  <motion.div
                    className="flex justify-start"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <div className="chat-bubble-ai">
                      <div className="flex items-center space-x-2">
                        <Sparkles className="h-4 w-4 text-blue-500" />
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <p className="text-sm">
                          {currentDocumentInfo ? "Gemini is analyzing the document..." : "Gemini is thinking..."}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          <div className="border-t p-4 bg-background/50 backdrop-blur-sm">
            <div className="flex space-x-2">
              <Textarea
                placeholder={
                  currentDocumentInfo
                    ? `Ask Gemini about ${currentDocumentInfo.filename}...`
                    : "Ask Gemini anything or upload a document for analysis..."
                }
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    if (!isProcessing && !isLoading) handleSendMessage()
                  }
                }}
                className="min-h-[60px] flex-1 resize-none rounded-xl border-muted-foreground/20 focus:border-primary"
                disabled={isProcessing || isLoading}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isProcessing || isLoading}
                className="h-auto rounded-xl shadow-md hover:shadow-lg transition-all duration-300"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}