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
    <div className="space-y-8">
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
              <p className="text-sm">
                Upload PDFs, DOCXs, or images and chat with their contents using OCR technology. Ask questions and get
                instant answers.
              </p>
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
              <p className="text-sm">
                Upload SQLite database files and query them using natural language instead of SQL. Get results in a
                readable format.
              </p>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/document-chat/sqlite" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300">Select</Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>

        {/* <motion.div variants={item}>
          <Card className="highlight-card h-full">
            <CardHeader>
              <div className="feature-icon mb-2">
                <BarChart className="h-6 w-6" />
              </div>
              <CardTitle>Data Visualization</CardTitle>
              <CardDescription>Visualize your data with charts</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm">
                Upload datasets in CSV or JSON format and generate customized visualizations. Create beautiful charts to
                understand your data.
              </p>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/document-chat/visualization" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300">Select</Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div> */}
      </motion.div>
    </div>
  )
}
