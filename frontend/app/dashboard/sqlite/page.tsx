"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Database, Upload, Send, X, Loader2, FileUp, History, Trash2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"

const API_BASE_URL = "http://localhost:8000"

interface DatabaseInfo {
  tables: string[]
  table_info: { [key: string]: string }
}

interface QueryResponse {
  response: string
  data?: any[]
  sql_query?: string
  timestamp: string
}

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  data?: any[]
  sql_query?: string
  timestamp: string
}

export default function SQLiteChatPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [sessionId, setSessionId] = useState<string>("")
  const [databaseInfo, setDatabaseInfo] = useState<DatabaseInfo | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [showHistory, setShowHistory] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  // Auto-scroll when messages change
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load chat history when session is established
  const loadChatHistory = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/sql-chat/chat-history/${sessionId}`)
      if (response.ok) {
        const data = await response.json()
        setMessages(data.messages || [])
      }
    } catch (error) {
      console.error("Failed to load chat history:", error)
    }
  }

  // Handle file upload
const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setIsProcessing(true)
    setUploadProgress(0)
    setMessages([]) // Clear previous messages

    try {
      // First, test if the API is reachable
      console.log("Testing API connection...")
      try {
        const testResponse = await fetch(`${API_BASE_URL}/api/sql-chat/debug`)
        const testText = await testResponse.text()
        console.log("Debug response status:", testResponse.status)
        console.log("Debug response:", testText)
        
        if (!testResponse.ok) {
          console.error("Debug endpoint failed - backend might not be running properly")
        }
      } catch (testError) {
        console.error("API connection test failed:", testError)
        throw new Error("Backend server is not responding. Please make sure the backend is running on port 8000.")
      }

      const formData = new FormData()
      formData.append("file", selectedFile)

      console.log("Uploading to:", `${API_BASE_URL}/api/sql-chat/upload-database`)
      console.log("File details:", {
        name: selectedFile.name,
        size: selectedFile.size,
        type: selectedFile.type
      })

      // Simulate upload progress
      const uploadInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 15, 90))
      }, 200)

      const response = await fetch(`${API_BASE_URL}/api/sql-chat/upload-database`, {
        method: "POST",
        body: formData,
      })

      clearInterval(uploadInterval)
      setUploadProgress(100)

      console.log("Upload response status:", response.status)
      console.log("Upload response headers:", Object.fromEntries(response.headers.entries()))

      if (!response.ok) {
        let errorMessage = "Upload failed"
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || `HTTP ${response.status}: ${response.statusText}`
          console.error("Upload error details:", errorData)
        } catch (parseError) {
          console.error("Failed to parse error response as JSON")
          try {
            const errorText = await response.text()
            console.error("Raw error response:", errorText)
            errorMessage = `HTTP ${response.status}: ${errorText || response.statusText}`
          } catch (textError) {
            console.error("Failed to get error text:", textError)
            errorMessage = `HTTP ${response.status}: ${response.statusText}`
          }
        }
        throw new Error(errorMessage)
      }

      const data = await response.json()
      console.log("Upload successful:", data)
      
      setSessionId(data.session_id)
      setDatabaseInfo(data.database_info)
      setIsProcessing(false)

      // Load any existing chat history for this session
      await loadChatHistory(data.session_id)

      toast({
        title: "Database loaded successfully",
        description: "You can now query your database using natural language with Llama 405B.",
        variant: "default",
      })
    } catch (error) {
      console.error("Upload error:", error)
      setIsProcessing(false)
      setFile(null)
      setUploadProgress(0)

      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload database",
        variant: "destructive",
      })
    }
  }

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !sessionId) return

    const userMessage: ChatMessage = {
      role: "user",
      content: inputMessage,
      timestamp: new Date().toISOString()
    }

    // Add user message immediately
    setMessages((prev) => [...prev, userMessage])
    setInputMessage("")
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/sql-chat/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: inputMessage,
          session_id: sessionId,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Query failed")
      }

      const data: QueryResponse = await response.json()

      const aiResponse: ChatMessage = {
        role: "assistant",
        content: data.response,
        data: data.data,
        sql_query: data.sql_query,
        timestamp: data.timestamp
      }

      setMessages((prev) => [...prev, aiResponse])
      setIsLoading(false)
    } catch (error) {
      console.error("Query error:", error)
      setIsLoading(false)

      const errorResponse: ChatMessage = {
        role: "assistant",
        content: `Error: ${error instanceof Error ? error.message : "Failed to process query"}`,
        timestamp: new Date().toISOString()
      }

      setMessages((prev) => [...prev, errorResponse])

      toast({
        title: "Query failed",
        description: error instanceof Error ? error.message : "Failed to process query",
        variant: "destructive",
      })
    }
  }

  // Clear the current database and session
  const handleClearDatabase = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_BASE_URL}/api/sql-chat/session/${sessionId}`, {
          method: "DELETE",
        })
      } catch (error) {
        console.error("Error clearing session:", error)
      }
    }

    setFile(null)
    setSessionId("")
    setDatabaseInfo(null)
    setMessages([])
    setInputMessage("")
    setUploadProgress(0)
    if (fileInputRef.current) fileInputRef.current.value = ""

    toast({
      title: "Database cleared",
      description: "Session has been cleared successfully.",
      variant: "default",
    })
  }

  // Clear chat history only
  const handleClearHistory = () => {
    setMessages([])
    toast({
      title: "Chat history cleared",
      description: "Local chat history has been cleared.",
      variant: "default",
    })
  }

  // Trigger file input click
  const triggerFileUpload = () => {
    fileInputRef.current?.click()
  }

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString()
    } catch {
      return ""
    }
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <h1 className="text-3xl font-bold tracking-tight gradient-text">Chat with SQLite</h1>
        <div className="text-muted-foreground mt-1 flex items-center gap-1 flex-wrap">
          <span>Upload a SQLite database and query it using natural language powered by</span>
          <Badge variant="secondary" className="ml-1">Llama 405B</Badge>
        </div>
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
                    <Database className="h-6 w-6" />
                  </div>
                  <div className="space-y-1 text-center">
                    <p className="text-sm font-medium">Upload a SQLite database</p>
                    <p className="text-xs text-muted-foreground">.db, .sqlite, or .sqlite3 files</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".db,.sqlite,.sqlite3"
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
                        <Database className="h-4 w-4 text-primary" />
                      </div>
                      <span className="text-sm font-medium">{file.name}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleClearDatabase}
                      title="Remove database"
                      className="rounded-full hover:bg-destructive/10 hover:text-destructive"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  {isProcessing ? (
                    <div className="space-y-3 p-4">
                      <div className="flex items-center justify-between text-sm">
                        <span>Processing database...</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} className="h-2" />
                      <div className="flex items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                      </div>
                    </div>
                  ) : (
                    <motion.div
                      className="space-y-4"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      {/* Session Info */}
                      <div className="rounded-md border p-3 bg-muted/30">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-sm font-medium">Session Active</h3>
                          <Badge variant="outline" className="text-xs">
                            {sessionId.substring(0, 8)}...
                          </Badge>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Powered by Llama 405B via NVIDIA NIM
                        </div>
                      </div>

                      {/* Database Schema */}
                      <div className="rounded-md border p-3">
                        <h3 className="mb-2 text-sm font-medium">Database Schema:</h3>
                        <div className="max-h-[300px] overflow-y-auto space-y-2">
                          <div className="text-xs rounded bg-muted/50 p-3">
                            <strong>Tables ({databaseInfo?.tables.length}):</strong>{" "}
                            {databaseInfo?.tables.join(", ")}
                          </div>
                          {databaseInfo?.table_info && Object.entries(databaseInfo.table_info).map(([table, info]) => (
                            <details key={table} className="text-xs">
                              <summary className="cursor-pointer font-medium hover:text-primary">
                                {table}
                              </summary>
                              <pre className="mt-1 whitespace-pre-wrap font-mono text-xs bg-muted/50 p-2 rounded">
                                {info}
                              </pre>
                            </details>
                          ))}
                        </div>
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
          {file && (
            <div className="flex items-center justify-between p-4 border-b bg-muted/30">
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-medium">SQL Chat</h3>
                <Badge variant="secondary" className="text-xs">
                  {messages.filter(m => m.role === "assistant").length} queries
                </Badge>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowHistory(!showHistory)}
                  className="text-xs"
                >
                  <History className="h-3 w-3 mr-1" />
                  History
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearHistory}
                  className="text-xs hover:text-destructive"
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <div className="feature-icon mb-4">
                  <Database className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-medium">Chat with your database</h3>
                <div className="mt-2 text-sm text-muted-foreground max-w-md">
                  Upload a SQLite database and ask questions using natural language. Our AI will translate your
                  questions into SQL queries using Llama 405B via NVIDIA NIM.
                </div>
                {!file && (
                  <Button
                    onClick={triggerFileUpload}
                    className="mt-4 gap-2 shadow-md hover:shadow-lg transition-all duration-300"
                  >
                    <FileUp className="h-4 w-4" />
                    Upload Database
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
                        <div className="flex items-start justify-between mb-1">
                          <div className="text-sm">{message.content}</div>
                          {showHistory && (
                            <span className="text-xs text-muted-foreground ml-2 shrink-0">
                              {formatTimestamp(message.timestamp)}
                            </span>
                          )}
                        </div>

                        {/* SQL Query Display */}
                        {message.sql_query && (
                          <motion.div
                            className="mt-2 p-2 bg-muted/50 rounded text-xs font-mono"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.1 }}
                          >
                            <div className="text-muted-foreground mb-1">SQL Query:</div>
                            <code className="text-primary">{message.sql_query}</code>
                          </motion.div>
                        )}

                        {/* Data Table Display */}
                        {message.data && message.data.length > 0 && (
                          <motion.div
                            className="mt-3 overflow-x-auto rounded border bg-background text-foreground"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.3, delay: 0.2 }}
                          >
                            <table className="min-w-full divide-y text-left text-sm">
                              <thead className="bg-muted/50">
                                <tr className="divide-x">
                                  {Object.keys(message.data[0]).map((key) => (
                                    <th key={key} className="px-3 py-2 font-medium">
                                      {key}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody className="divide-y">
                                {message.data.slice(0, 10).map((row, rowIndex) => (
                                  <tr key={rowIndex} className="divide-x hover:bg-muted/30 transition-colors">
                                    {Object.values(row).map((value, cellIndex) => (
                                      <td key={cellIndex} className="px-3 py-2">
                                        {String(value)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {message.data.length > 10 && (
                              <div className="p-2 text-xs text-muted-foreground text-center border-t">
                                Showing 10 of {message.data.length} rows
                              </div>
                            )}
                          </motion.div>
                        )}
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
                        <div className="text-sm">Llama 405B is thinking...</div>
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
                placeholder={file ? "Ask a question about your database..." : "Upload a database to start querying..."}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    if (file && !isProcessing && !isLoading) handleSendMessage()
                  }
                }}
                className="min-h-[60px] flex-1 resize-none rounded-xl border-muted-foreground/20 focus:border-primary"
                disabled={!file || isProcessing || isLoading}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!file || !inputMessage.trim() || isProcessing || isLoading}
                className="h-auto rounded-xl shadow-md hover:shadow-lg transition-all duration-300"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
            {file && (
              <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>Press Enter to send, Shift+Enter for new line</span>
                <span>Session: {sessionId.substring(0, 8)}...</span>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  )
}