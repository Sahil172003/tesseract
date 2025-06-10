"use client"

import type React from "react"

import { useState } from "react"
import { BarChart, Upload, Send, X, Loader2, PieChart, LineChart, BarChart2, Image as ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Bar,
  BarChart as RechartsBarChart,
  Line,
  LineChart as RechartsLineChart,
  Pie,
  PieChart as RechartsPieChart,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042"]

interface ChartData {
  type: "bar" | "line" | "pie"
  data: Array<{ name: string; value: number }>
}

interface Message {
  role: "user" | "assistant"
  content: string
  chartData?: ChartData
  visualizations?: string[]
}

export default function VisualizationPage() {
  const { toast } = useToast()
  const [file, setFile] = useState<File | null>(null)
  const [dataPreview, setDataPreview] = useState<string>("")
  const [chartType, setChartType] = useState<"bar" | "line" | "pie">("bar")
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [summaryStats, setSummaryStats] = useState<any>(null)

  // Handle file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setIsProcessing(true)

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
        setDataPreview(result.data_preview || "Data loaded successfully")
        setSummaryStats(result.summary_stats || {})
        
        toast({
          title: "Data loaded successfully",
          description: "You can now visualize your data.",
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

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !file) return

    // Add user message
    const userMessage: Message = { role: "user", content: inputMessage }
    setMessages(prev => [...prev, userMessage])
    const currentInput = inputMessage
    setInputMessage("")

    // Show loading
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/data-visualization/visualize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visualization_request: currentInput,
          chart_type: chartType
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`)
      }

      const result = await response.json()
      
      if (result.status === 'success') {
        const aiResponse: Message = {
          role: "assistant",
          content: result.narrative || "I've analyzed your data and created visualizations based on your request.",
          chartData: result.chart_data,
          visualizations: result.visualizations
        }
        
        setMessages(prev => [...prev, aiResponse])
      } else {
        throw new Error(result.message || 'Visualization failed')
      }
    } catch (error) {
      console.error('Visualization error:', error)
      const errorMessage: Message = {
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
      setMessages(prev => [...prev, errorMessage])
      
      toast({
        title: "Visualization failed",
        description: error instanceof Error ? error.message : "Failed to create visualization",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
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
    setDataPreview("")
    setMessages([])
    setInputMessage("")
    setSummaryStats(null)
  }

  // Render chart based on type
  const renderChart = (chartData: ChartData) => {
    if (chartData.type === "bar") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <RechartsBarChart data={chartData.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#8884d8" />
          </RechartsBarChart>
        </ResponsiveContainer>
      )
    } else if (chartData.type === "line") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <RechartsLineChart data={chartData.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </RechartsLineChart>
        </ResponsiveContainer>
      )
    } else {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <RechartsPieChart>
            <Pie
              data={chartData.data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {chartData.data.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </RechartsPieChart>
        </ResponsiveContainer>
      )
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Visualization</h1>
        <p className="text-muted-foreground">Upload data and create visualizations using natural language.</p>
      </div>

      <div className="grid gap-6 md:grid-cols-[1fr_2fr]">
        <div className="space-y-4">
          <Card>
            <CardContent className="p-4">
              {!file ? (
                <div className="flex flex-col items-center justify-center space-y-2 rounded-md border border-dashed p-8">
                  <BarChart className="h-8 w-8 text-muted-foreground" />
                  <div className="space-y-1 text-center">
                    <p className="text-sm font-medium">Upload data</p>
                    <p className="text-xs text-muted-foreground">CSV or Excel files</p>
                  </div>
                  <label htmlFor="file-upload">
                    <div className="mt-2 flex cursor-pointer items-center justify-center rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground shadow-sm">
                      <Upload className="mr-2 h-4 w-4" />
                      Upload
                    </div>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept=".csv,.xlsx,.xls"
                      onChange={handleFileUpload}
                    />
                  </label>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <BarChart className="h-5 w-5 text-muted-foreground" />
                      <span className="text-sm font-medium">{file.name}</span>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handleClearData} title="Remove data">
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  {isProcessing ? (
                    <div className="flex items-center justify-center p-4">
                      <Loader2 className="h-6 w-6 animate-spin text-primary" />
                      <span className="ml-2 text-sm">Processing data...</span>
                    </div>
                  ) : (
                    <>
                      <div className="rounded-md border p-3">
                        <h3 className="mb-2 text-sm font-medium">Data Preview:</h3>
                        <pre className="max-h-[200px] overflow-y-auto whitespace-pre-wrap text-xs">{dataPreview}</pre>
                      </div>

                      {summaryStats && Object.keys(summaryStats).length > 0 && (
                        <div className="rounded-md border p-3">
                          <h3 className="mb-2 text-sm font-medium">Summary Statistics:</h3>
                          <div className="text-xs">
                            <p>Numeric columns: {Object.keys(summaryStats).length}</p>
                          </div>
                        </div>
                      )}

                      <div className="space-y-2">
                        <h3 className="text-sm font-medium">Chart Type:</h3>
                        <Tabs
                          defaultValue="bar"
                          className="w-full"
                          onValueChange={(value) => setChartType(value as "bar" | "line" | "pie")}
                        >
                          <TabsList className="grid w-full grid-cols-3">
                            <TabsTrigger value="bar" className="flex items-center gap-1">
                              <BarChart2 className="h-4 w-4" />
                              <span>Bar</span>
                            </TabsTrigger>
                            <TabsTrigger value="line" className="flex items-center gap-1">
                              <LineChart className="h-4 w-4" />
                              <span>Line</span>
                            </TabsTrigger>
                            <TabsTrigger value="pie" className="flex items-center gap-1">
                              <PieChart className="h-4 w-4" />
                              <span>Pie</span>
                            </TabsTrigger>
                          </TabsList>
                        </Tabs>
                      </div>
                    </>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="flex h-[600px] flex-col rounded-md border">
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center text-center">
                <BarChart className="h-12 w-12 text-muted-foreground" />
                <h3 className="mt-4 text-lg font-medium">Visualize your data</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Upload data and describe the visualization you want to create.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                      }`}
                    >
                      <div className="text-sm whitespace-pre-wrap">{message.content}</div>

                      {message.chartData && (
                        <div className="mt-4 rounded border bg-background p-2 text-foreground">
                          {renderChart(message.chartData)}
                        </div>
                      )}

                      {message.visualizations && message.visualizations.length > 0 && (
                        <div className="mt-4 space-y-2">
                          <div className="flex items-center gap-2 text-sm font-medium">
                            <ImageIcon className="h-4 w-4" />
                            Generated Visualizations:
                          </div>
                          <div className="grid gap-2">
                            {message.visualizations.map((viz, vizIndex) => (
                              <div key={vizIndex} className="rounded border bg-background p-2">
                                <img 
                                  src={viz} 
                                  alt={`Visualization ${vizIndex + 1}`}
                                  className="w-full h-auto rounded"
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] rounded-lg bg-muted px-4 py-2">
                      <div className="flex items-center space-x-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <p className="text-sm">Generating visualization...</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="border-t p-4">
            <div className="flex space-x-2">
              <Textarea
                placeholder={file ? "Describe the visualization you want..." : "Upload data to start visualizing..."}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    if (file) handleSendMessage()
                  }
                }}
                className="min-h-[60px] flex-1 resize-none"
                disabled={!file || isProcessing}
              />
              <Button
                onClick={handleSendMessage}
                disabled={!file || !inputMessage.trim() || isProcessing || isLoading}
                className="h-auto"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}