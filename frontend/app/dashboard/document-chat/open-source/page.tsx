"use client"

import type React from "react"

import { useState, useRef } from "react"
import { FileText, Upload, Send, X, Loader2, FileUp } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Progress } from "@/components/ui/progress"

export default function OCRChatPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [extractedText, setExtractedText] = useState<string>("")
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setIsProcessing(true)
    setUploadProgress(0)

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 5
      })
    }, 100)

    // Simulate OCR processing
    setTimeout(() => {
      clearInterval(interval)
      setUploadProgress(100)

      // Mock extracted text
      // TODO : Add actual working API
      const mockExtractedText =
        "This is a sample extracted text from the document. It would contain the actual content of your uploaded document after OCR processing. You can now ask questions about this content."
      setExtractedText(mockExtractedText)
      setIsProcessing(false)

      toast({
        title: "Document processed successfully",
        description: "You can now chat with your document.",
        variant: "success",
      })
    }, 2000)
  }

  // Handle sending a message
  const handleSendMessage = () => {
    if (!inputMessage.trim()) return

    // Add user message
    const userMessage = { role: "user" as const, content: inputMessage }
    setMessages((prev) => [...prev, userMessage])
    setInputMessage("")

    // Simulate AI response
    setIsLoading(true)
    setTimeout(() => {
      const aiResponse = {
        role: "assistant" as const,
        content: `I've analyzed the document and found information related to your query. ${Math.random() > 0.5 ? "The document mentions specific details about this topic." : "Based on the extracted text, I can provide the following information."}`,
      }
      setMessages((prev) => [...prev, aiResponse])
      setIsLoading(false)

      // Scroll to bottom after message is added
      setTimeout(scrollToBottom, 100)
    }, 1500)
  }

  // Clear the current document
  const handleClearDocument = () => {
    setFile(null)
    setExtractedText("")
    setMessages([])
    setInputMessage("")
    setUploadProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  // Trigger file input click
  const triggerFileUpload = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <h1 className="text-3xl font-bold tracking-tight gradient-text">Chat with Document</h1>
        <p className="text-muted-foreground mt-1">Upload a document and chat with its contents using OCR technology.</p>
      </motion.div>

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <motion.div
          className="space-y-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
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
                    <p className="text-sm font-medium">Upload a document</p>
                    <p className="text-xs text-muted-foreground">PDF, DOCX, or image files</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".pdf,.docx,.png,.jpg,.jpeg"
                    onChange={handleFileUpload}
                  />
                  <Button
                    onClick={triggerFileUpload}
                    className="gap-2 shadow-md hover:shadow-lg transition-all duration-300"
                  >
                    <Upload className="h-4 w-4" />
                    Upload
                  </Button>
                </motion.div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                        <FileText className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-sm font-medium">{file.name}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleClearDocument}
                      title="Remove document"
                      className="rounded-full hover:bg-destructive/10 hover:text-destructive"
                    >
                      <X className="h-4 w-4" />
                    </Button>
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
                    </div>
                  ) : (
                    <motion.div
                      className="rounded-md border p-3"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <h3 className="mb-2 text-sm font-medium">Extracted Text:</h3>
                      <div className="max-h-[300px] overflow-y-auto text-sm rounded bg-muted/50 p-3">
                        {extractedText}
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
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <div className="feature-icon mb-4">
                  <FileText className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-medium">Chat with your document</h3>
                <p className="mt-2 text-sm text-muted-foreground max-w-md">
                  Upload a document and ask questions about its contents. Our AI will analyze the document and provide
                  answers.
                </p>
                {!file && (
                  <Button
                    onClick={triggerFileUpload}
                    className="mt-4 gap-2 shadow-md hover:shadow-lg transition-all duration-300"
                  >
                    <FileUp className="h-4 w-4" />
                    Upload Document
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
                        <p className="text-sm">{message.content}</p>
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
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <p className="text-sm">Thinking...</p>
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
                placeholder={file ? "Ask a question about your document..." : "Upload a document to start chatting..."}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    if (file && !isProcessing) handleSendMessage()
                  }
                }}
                className="min-h-[60px] flex-1 resize-none rounded-xl border-muted-foreground/20 focus:border-primary"
                disabled={!file || isProcessing}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!file || !inputMessage.trim() || isProcessing}
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
