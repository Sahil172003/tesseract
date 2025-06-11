"use client"

import Link from "next/link"
import { FileText, Database, BarChart } from "lucide-react"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export default function DocumentChatPage() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <h1 className="text-4xl font-bold tracking-tight gradient-text">OCR Chat</h1>
        <p className="text-muted-foreground mt-2">Choose an option below to start chatting with your data.</p>
      </motion.div>

      <motion.div className="grid gap-6 md:grid-cols-2" variants={container} initial="hidden" animate="show">
        <motion.div variants={item}>
          <Card className="highlight-card h-full">
            <CardHeader>
              <div className="feature-icon mb-2">
                <FileText className="h-6 w-6" />
              </div>
              <CardTitle>Chat with Legacy Model</CardTitle>
              <CardDescription>Upload and chat with any document</CardDescription>
            </CardHeader>
            <CardContent>
              <div className=" text-purple-100 p-6 rounded-lg shadow-lg">
  {/* Legacy Models */}
  <div className="mb-6">
    <h2 className="text-2xl font-bold text-purple-300 mb-4">Legacy Models (e.g., Gemini, Claude)</h2>
    <ul className="list-disc pl-5 space-y-2">
      <li>Closed-source, accessible via APIs or paid subscriptions.</li>
      <li>Data privacy controlled by the provider.</li>
      <li>Often integrated into commercial platforms for enterprise use cases.</li>
      <li>Regular updates managed by the provider, ensuring stability.</li>
      <li>Designed with strict safety and moderation filters for public use.</li>
      <li>Typically require internet connectivity for API access.</li>
    </ul>
  </div>
</div>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/document-chat/ocr" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300">Select</Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="highlight-card h-full">
            <CardHeader>
              <div className="feature-icon mb-2">
                <Database className="h-6 w-6" />
              </div>
              <CardTitle>Chat with Open-Source Model</CardTitle>
              <CardDescription>Query databases with natural language</CardDescription>
            </CardHeader>
            <CardContent>
              <div className=" text-purple-100 p-6 rounded-lg shadow-lg">
    <h2 className="text-2xl font-bold text-purple-300 mb-4">Open-Source Models (e.g., NVIDIA’s models)</h2>
    <ul className="list-disc pl-5 space-y-2">
      <li>Publicly available models from companies like NVIDIA or community efforts.</li>
      <li>Performance varies; can be high with proper fine-tuning.</li>
      <li>Outputs depend on community contributions and training data.</li>
      <li>Highly customizable; users can modify weights and architecture.</li>
      <li>Support a wide range of hardware, including GPUs and TPUs.</li>
      <li>Community-driven documentation and tutorials available.</li>
    </ul>
  </div>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/document-chat/open-source" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300">Select</Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  )
}
