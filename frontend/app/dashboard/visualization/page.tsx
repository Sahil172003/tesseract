"use client"

import type React from "react"

import { useState } from "react"
import { BarChart, Upload, Send, X, Loader2, MessageCircle, Sparkles, Image as ImageIcon, Code, Table } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { useToast } from "@/hooks/use-toast"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"

interface TableData {
  columns: string[]
  data: Array<Record<string, any>>
  total_rows: number
  total_columns: number
}

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  code_executed?: string
  execution_result?: string
  timestamp: string
}

interface VisualizationMessage {
  role: "user" | "assistant"
  content: string
  visualizations?: string[]  // Changed to simple string array since we get base64 directly
  timestamp: string
}

export default function VisualizationPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [tableData, setTableData] = useState<TableData | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isCreatingViz, setIsCreatingViz] = useState(false)
  const [isChatLoading, setIsChatLoading] = useState(false)
  
  // Visualization section
  const [vizRequest, setVizRequest] = useState("")
  const [vizMessages, setVizMessages] = useState<VisualizationMessage[]>([])
  
  // Chat section
  const [chatQuestion, setChatQuestion] = useState("")
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setIsProcessing(true)
    setTableData(null)
    setVizMessages([])
    setChatMessages([])

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://localhost:8000/api/data-visualization/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (result.status === 'success') {
        setTableData(result.table_data)
        
        toast({
          title: "Data loaded successfully",
          description: `Loaded ${result.table_data.total_rows} rows and ${result.table_data.total_columns} columns`,
        })
      } else {
        throw new Error(result.message || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload error:', error)
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload file",
        variant: "destructive",
      })
      setFile(null)
    } finally {
      setIsProcessing(false)
    }
  }

  // Handle create visualization
  const handleCreateVisualization = async () => {
    if (!file) return

    setIsCreatingViz(true)
    
    const userMessage: VisualizationMessage = {
      role: "user",
      content: vizRequest || "Create comprehensive visualizations for this dataset",
      timestamp: new Date().toLocaleTimeString()
    }
    setVizMessages(prev => [...prev, userMessage])
    const currentRequest = vizRequest
    setVizRequest("")

    try {
      const response = await fetch('http://localhost:8000/api/data-visualization/create-visualization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visualization_request: currentRequest || "Create comprehensive visualizations for this dataset"
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (result.status === 'success') {
        const aiResponse: VisualizationMessage = {
          role: "assistant",
          content: "I've created comprehensive visualizations for your dataset based on the data structure and characteristics.",
          visualizations: result.visualizations,
          timestamp: new Date().toLocaleTimeString()
        }
        
        setVizMessages(prev => [...prev, aiResponse])
        
        toast({
          title: "Visualizations created",
          description: `Generated ${result.visualizations?.length || 0} visualizations`,
        })
      } else {
        throw new Error(result.message || 'Visualization failed')
      }
    } catch (error) {
      console.error('Visualization error:', error)
      const errorMessage: VisualizationMessage = {
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toLocaleTimeString()
      }
      setVizMessages(prev => [...prev, errorMessage])
      
      toast({
        title: "Visualization failed",
        description: error instanceof Error ? error.message : "Failed to create visualization",
        variant: "destructive",
      })
    } finally {
      setIsCreatingViz(false)
    }
  }

  // Handle chat question
  const handleChatQuestion = async () => {
    if (!chatQuestion.trim() || !file) return

    setIsChatLoading(true)
    
    const userMessage: ChatMessage = {
      role: "user",
      content: chatQuestion,
      timestamp: new Date().toLocaleTimeString()
    }
    setChatMessages(prev => [...prev, userMessage])
    const currentQuestion = chatQuestion
    setChatQuestion("")

    try {
      const response = await fetch('http://localhost:8000/api/data-visualization/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`)
      }

      const result = await response.json()
      
      const aiResponse: ChatMessage = {
        role: "assistant",
        content: result.response,
        code_executed: result.code_executed,
        execution_result: result.execution_result,
        timestamp: new Date().toLocaleTimeString()
      }
      
      setChatMessages(prev => [...prev, aiResponse])
      
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toLocaleTimeString()
      }
      setChatMessages(prev => [...prev, errorMessage])
    } finally {
      setIsChatLoading(false)
    }
  }

  // Clear the current data
  const handleClearData = async () => {
    try {
      await fetch('http://localhost:8000/api/data-visualization/clear', {
        method: 'POST',
      })
    } catch (error) {
      console.error('Clear error:', error)
    }
    
    setFile(null)
    setTableData(null)
    setVizMessages([])
    setChatMessages([])
    setVizRequest("")
    setChatQuestion("")
  }

  return (
    <div className="new min-h-screen w-full max-w-none p-2 sm:p-4 md:p-6 space-y-4 md:space-y-6 overflow-x-hidden">
      <div className="w-full">
        <h1 className="text-xl sm:text-2xl md:text-3xl font-bold tracking-tight break-words">Data Visualization & Analysis</h1>
        <p className="text-sm md:text-base text-muted-foreground mt-1">Upload data, create visualizations, and ask questions about your dataset.</p>
      </div>

      {/* File Upload Section */}
      <Card className="w-full max-w-full">
        <CardContent className="p-3 sm:p-4 md:p-6">
          {!file ? (
            <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg">
              <div className="flex flex-col items-center justify-center p-6 md:p-8 text-center">
                <Upload className="h-8 w-8 sm:h-10 sm:w-10 text-muted-foreground mb-4" />
                <div className="space-y-2">
                  <h3 className="text-base sm:text-lg font-medium">Upload your dataset</h3>
                  <p className="text-sm text-muted-foreground max-w-md">
                    Choose a CSV, Excel, or JSON file to start analyzing your data
                  </p>
                </div>
                <div className="mt-4">
                  <input
                    type="file"
                    id="file-upload"
                    accept=".csv,.xlsx,.xls,.json"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button asChild className="w-full sm:w-auto">
                    <label htmlFor="file-upload" className="cursor-pointer">
                      <Upload className="h-4 w-4 mr-2" />
                      Choose File
                    </label>
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4 w-full">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                <div className="flex items-center space-x-3 min-w-0 flex-1">
                  <Table className="h-5 w-5 text-green-600 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="font-medium truncate text-sm md:text-base">{file.name}</p>
                    {tableData && (
                      <p className="text-xs md:text-sm text-muted-foreground">
                        {tableData.total_rows} rows × {tableData.total_columns} columns
                      </p>
                    )}
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={handleClearData} title="Remove data" className="flex-shrink-0">
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {isProcessing ? (
                <div className="flex items-center justify-center p-6 md:p-8">
                  <Loader2 className="h-6 w-6 animate-spin text-primary" />
                  <span className="ml-2 text-sm md:text-base">Processing data...</span>
                </div>
              ) : tableData && (
                <div className="w-full">
                  <div className="rounded-md border w-full overflow-hidden">
                    <div className="border-b bg-muted/50 px-3 md:px-4 py-2">
                      <h3 className="font-medium text-sm md:text-base">Data Preview</h3>
                    </div>
                    <div className="w-full">
                      <ScrollArea className="w-full">
                        <div className="p-3 md:p-4 w-full min-w-0">
                          <div className="w-full overflow-x-auto">
                            <table className="w-full text-xs sm:text-sm border-collapse">
                              <thead>
                                <tr className="border-b">
                                  {tableData.columns.map((column, index) => (
                                    <th key={index} className="text-left p-1 md:p-2 font-medium whitespace-nowrap min-w-[80px] sm:min-w-[100px]">
                                      {column}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {tableData.data.slice(0, 10).map((row, rowIndex) => (
                                  <tr key={rowIndex} className="border-b hover:bg-muted/50">
                                    {tableData.columns.map((column, colIndex) => (
                                      <td key={colIndex} className="p-1 md:p-2 whitespace-nowrap min-w-[80px] sm:min-w-[100px] max-w-[200px] truncate">
                                        {row[column]?.toString() || ''}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </ScrollArea>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Content - Only show if file is uploaded */}
      {tableData && (
        <div className="w-full space-y-4 md:space-y-6 xl:space-y-0 xl:grid xl:grid-cols-2 xl:gap-4 2xl:gap-6">
          {/* Visualization Section */}
          <Card className="w-full max-w-full">
            <CardHeader className="pb-3 px-3 md:px-6">
              <CardTitle className="flex items-center gap-2 text-base md:text-lg">
                <Sparkles className="h-4 w-4 md:h-5 md:w-5 flex-shrink-0" />
                <span className="truncate">Create Visualizations</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {/* Visualization Input */}
              <div className="p-3 md:p-4 border-b">
                <div className="flex flex-col sm:flex-row gap-2">
                  <Input
                    placeholder="Describe what visualizations you want (optional)..."
                    value={vizRequest}
                    onChange={(e) => setVizRequest(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault()
                        handleCreateVisualization()
                      }
                    }}
                    className="flex-1 min-w-0 text-sm"
                  />
                  <Button
                    onClick={handleCreateVisualization}
                    disabled={isCreatingViz}
                    className="shrink-0 w-full sm:w-auto"
                    size="sm"
                  >
                    {isCreatingViz ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4" />
                    )}
                    <span className="ml-2">Create</span>
                  </Button>
                </div>
              </div>

              {/* Visualization Results */}
              <div className="w-full">
                <ScrollArea className="h-[300px] sm:h-[400px] lg:h-[500px] w-full">
                  <div className="p-3 md:p-4 w-full min-w-0">
                    {vizMessages.length === 0 ? (
                      <div className="flex h-[250px] sm:h-[350px] lg:h-[450px] flex-col items-center justify-center text-center text-muted-foreground">
                        <Sparkles className="h-8 w-8 sm:h-12 sm:w-12 mb-4" />
                        <p className="text-base sm:text-lg font-medium mb-2">Ready to visualize!</p>
                        <p className="text-sm max-w-md px-4">Click "Create" to generate comprehensive visualizations for your dataset.</p>
                      </div>
                    ) : (
                      <div className="space-y-4 w-full">
                        {vizMessages.map((message, index) => (
                          <div key={index} className={`flex w-full ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                            <div className={`max-w-[95%] sm:max-w-[90%] rounded-lg p-3 break-words ${
                              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                            }`}>
                              <div className="text-sm mb-1 whitespace-pre-wrap break-words">{message.content}</div>
                              <div className="text-xs opacity-70">{message.timestamp}</div>
                              
                             {message.visualizations && message.visualizations.length > 0 && (
                                <div className="mt-3 space-y-3 w-full">
                                  {message.visualizations.map((vizBase64, vizIndex) => (
                                    <div key={vizIndex} className="bg-background/10 rounded p-2 w-full">
                                      <div className="flex items-center gap-2 mb-2">
                                        <ImageIcon className="h-4 w-4" />
                                        <span className="text-xs font-medium">Visualization {vizIndex + 1}</span>
                                      </div>
                                      <div className="w-full overflow-hidden rounded">
                                        <img 
                                          src={vizBase64} 
                                          alt={`Visualization ${vizIndex + 1}`}
                                          className="w-full h-auto max-w-full object-contain"
                                          style={{ maxHeight: '400px' }}
                                        />
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {isCreatingViz && index === vizMessages.length - 1 && (
                                <div className="mt-2 flex items-center gap-2">
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                  <p className="text-sm">Generating visualizations...</p>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </div>
            </CardContent>
          </Card>

          {/* Chat Section */}
          <Card className="w-full max-w-full">
            <CardHeader className="pb-3 px-3 md:px-6">
              <CardTitle className="flex items-center gap-2 text-base md:text-lg">
                <MessageCircle className="h-4 w-4 md:h-5 md:w-5 flex-shrink-0" />
                <span className="truncate">Ask Questions</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {/* Chat Messages */}
              <div className="w-full">
                <ScrollArea className="h-[300px] sm:h-[400px] lg:h-[450px] w-full">
                  <div className="p-3 md:p-4 w-full min-w-0">
                    {chatMessages.length === 0 ? (
                      <div className="flex h-[250px] sm:h-[350px] lg:h-[400px] flex-col items-center justify-center text-center text-muted-foreground">
                        <MessageCircle className="h-8 w-8 sm:h-12 sm:w-12 mb-4" />
                        <p className="text-base sm:text-lg font-medium mb-2">Ask about your data</p>
                        <p className="text-sm max-w-md px-4">Ask questions like "How many rows?", "What are the columns?", or "Show me summary statistics"</p>
                      </div>
                    ) : (
                      <div className="space-y-4 w-full">
                        {chatMessages.map((message, index) => (
                          <div key={index} className={`flex w-full ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                            <div className={`max-w-[95%] sm:max-w-[90%] rounded-lg p-3 break-words ${
                              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                            }`}>
                              <div className="text-sm mb-1 whitespace-pre-wrap break-words">{message.content}</div>
                              <div className="text-xs opacity-70">{message.timestamp}</div>
                              
                              {message.code_executed && (
                                <div className="mt-2 p-2 bg-background/10 rounded text-xs w-full">
                                  <div className="flex items-center gap-1 mb-1">
                                    <Code className="h-3 w-3" />
                                    <span className="font-medium">Code executed</span>
                                  </div>
                                  <pre className="whitespace-pre-wrap break-words overflow-x-auto text-xs">{message.code_executed}</pre>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </div>

              {/* Chat Input */}
              <div className="border-t p-3 md:p-4">
                <div className="flex flex-col sm:flex-row gap-2">
                  <Input
                    placeholder="Ask a question about your data..."
                    value={chatQuestion}
                    onChange={(e) => setChatQuestion(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault()
                        handleChatQuestion()
                      }
                    }}
                    className="flex-1 min-w-0 text-sm"
                  />
                  <Button
                    onClick={handleChatQuestion}
                    disabled={!chatQuestion.trim() || isChatLoading}
                    size="icon"
                    className="shrink-0 w-full sm:w-auto"
                  >
                    <Send className="h-4 w-4" />
                    <span className="ml-2 sm:hidden">Send</span>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}