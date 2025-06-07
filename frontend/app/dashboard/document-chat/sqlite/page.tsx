"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Database, Upload, Send, X, Loader2, FileUp } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Progress } from "@/components/ui/progress"

export default function SQLiteChatPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [databaseInfo, setDatabaseInfo] = useState<string>("")
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string; data?: any[] }[]>([])
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

    // Simulate database processing
    setTimeout(() => {
      clearInterval(interval)
      setUploadProgress(100)

      // Mock database info
      // TODO : Add actual working API
      const mockDatabaseInfo =
        "Database contains the following tables:\n- users (id, name, email, created_at)\n- products (id, name, price, category_id)\n- categories (id, name)\n- orders (id, user_id, total, status, created_at)"
      setDatabaseInfo(mockDatabaseInfo)
      setIsProcessing(false)

      toast({
        title: "Database loaded successfully",
        description: "You can now query your database using natural language.",
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
      // Generate mock data for table results
      const mockData =
        Math.random() > 0.5
          ? [
              { id: 1, name: "John Doe", email: "john@example.com" },
              { id: 2, name: "Jane Smith", email: "jane@example.com" },
              { id: 3, name: "Bob Johnson", email: "bob@example.com" },
            ]
          : null

      const aiResponse = {
        role: "assistant" as const,
        content: `I've translated your query to SQL and executed it. ${mockData ? "Here are the results:" : "The query was executed successfully, but returned no results."}`,
        data: mockData,
      }
      setMessages((prev) => [...prev, aiResponse])
      setIsLoading(false)

      // Scroll to bottom after message is added
      setTimeout(scrollToBottom, 100)
    }, 1500)
  }

  // Clear the current database
  const handleClearDatabase = () => {
    setFile(null)
    setDatabaseInfo("")
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
        <h1 className="text-3xl font-bold tracking-tight gradient-text">Chat with SQLite</h1>
        <p className="text-muted-foreground mt-1">Upload a SQLite database and query it using natural language.</p>
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
                    <p className="text-xs text-muted-foreground">.db or .sqlite files</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".db,.sqlite"
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
                      className="rounded-md border p-3"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.5 }}
                    >
                      <h3 className="mb-2 text-sm font-medium">Database Schema:</h3>
                      <pre className="max-h-[300px] overflow-y-auto whitespace-pre-wrap text-xs rounded bg-muted/50 p-3 font-mono">
                        {databaseInfo}
                      </pre>
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
                  <Database className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-medium">Chat with your database</h3>
                <p className="mt-2 text-sm text-muted-foreground max-w-md">
                  Upload a SQLite database and ask questions using natural language. Our AI will translate your
                  questions into SQL queries.
                </p>
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
                        <p className="text-sm">{message.content}</p>

                        {message.data && (
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
                                {message.data.map((row, rowIndex) => (
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
                placeholder={file ? "Ask a question about your database..." : "Upload a database to start querying..."}
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
