"use client"

import Link from "next/link"
import { FileText, ImageIcon, Sparkles, ArrowRight, BarChart3, Clock, ChevronRight } from "lucide-react"
import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export default function DashboardPage() {
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
    <div className="space-y-10">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold tracking-tight gradient-text">Dashboard</h1>
            <p className="text-muted-foreground mt-2 text-lg">Welcome to the GenAI App. Choose an option below to get started.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="gap-1 transition-all duration-200">
              <Clock className="h-4 w-4" />
              Activity
            </Button>
            <Button variant="outline" size="sm" className="gap-1 transition-all duration-200">
              <BarChart3 className="h-4 w-4" />
              Statistics
            </Button>
          </div>
        </div>
      </motion.div>

      <motion.div
        className="relative overflow-hidden rounded-xl border p-8 shadow-md bg-gradient-to-r from-primary/5 to-primary/10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className="absolute top-0 right-0 -mt-4 -mr-4 h-24 w-24 rounded-full bg-primary/20 blur-xl"></div>
        <div className="absolute bottom-0 left-0 -mb-4 -ml-4 h-24 w-24 rounded-full bg-primary/20 blur-xl"></div>

        <div className="relative flex flex-col md:flex-row gap-8 items-center">
          <div className="flex-1 space-y-4">
            <Badge variant="outline" className="bg-primary/10 hover:bg-primary/20 transition-colors border-0 text-primary px-3 py-1">
              Quick Start
            </Badge>
            <h2 className="text-3xl font-bold">Ready to explore AI capabilities?</h2>
            <p className="text-muted-foreground text-lg max-w-[600px]">
              Get started with document chat or image generation. Our AI-powered tools help you work smarter, not
              harder.
            </p>
          </div>
          <div className="flex gap-4">
            <Link href="/dashboard/document-chat" passHref>
              <Button className="gap-2 px-6 py-6 h-auto shadow-lg hover:shadow-primary/25 transition-all duration-300 hover:translate-y-[-2px]">
                Try Document Chat
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </motion.div>

      <motion.div className="grid gap-8 md:grid-cols-4 xl:grid-cols-4" variants={container} initial="hidden" animate="show">
        <motion.div variants={item}>
          <Card className="highlight-card h-full border border-border/50 transition-all duration-300 hover:border-primary/20 hover:shadow-lg">
            <CardHeader className="pb-4">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 transition-all duration-300 group-hover:bg-primary/20">
                <FileText className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl">Document Chat</CardTitle>
              <CardDescription className="text-base">
                Chat with documents, databases, and visualize data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Upload and chat with any document using OCR</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Query databases using natural language</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Create beautiful visualizations from your data</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/document-chat" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300 hover:translate-y-[-2px] group">
                  Get Started
                  <ChevronRight className="h-4 w-4 ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                </Button>
              </Link>

            </CardFooter>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="highlight-card h-full border border-border/50 transition-all duration-300 hover:border-primary/20 hover:shadow-lg">
            <CardHeader className="pb-4">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 transition-all duration-300 group-hover:bg-primary/20">
                <ImageIcon className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl">Chat with Sql</CardTitle>
              <CardDescription className="text-base">Generate stunning images with AI</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Create images from text descriptions</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Customize style, resolution, and more</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Download and share your creations</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/sqlite" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300 hover:translate-y-[-2px] group">
                  Get Started
                  <ChevronRight className="h-4 w-4 ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                </Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>

        <motion.div variants={item} className="md:col-span-2 xl:col-span-1">
          <Card className="highlight-card h-full border border-border/50 transition-all duration-300 hover:border-primary/20 hover:shadow-lg">
            <CardHeader className="pb-4">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 transition-all duration-300 group-hover:bg-primary/20">
                <BarChart3 className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl">Data Analysis</CardTitle>
              <CardDescription className="text-base">
                Analyze and visualize complex data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Create interactive data visualizations</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Get AI-powered insights from your data</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Export charts and reports in multiple formats</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/visualization" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300 hover:translate-y-[-2px] group">
                  Get Started
                  <ChevronRight className="h-4 w-4 ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                </Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>
        <motion.div variants={item}>
          <Card className="highlight-card h-full border border-border/50 transition-all duration-300 hover:border-primary/20 hover:shadow-lg">
            <CardHeader className="pb-4">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 transition-all duration-300 group-hover:bg-primary/20">
                <ImageIcon className="h-6 w-6 text-primary" />
              </div>
              <CardTitle className="text-2xl">Image Generation</CardTitle>
              <CardDescription className="text-base">Generate stunning images with AI</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Create images from text descriptions</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Customize style, resolution, and more</span>
                </li>
                <li className="flex items-start gap-2">
                  <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                  <span>Download and share your creations</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Link href="/dashboard/image-generation" passHref className="w-full">
                <Button className="w-full shadow-md hover:shadow-lg transition-all duration-300 hover:translate-y-[-2px] group">
                  Get Started
                  <ChevronRight className="h-4 w-4 ml-1 transition-transform duration-300 group-hover:translate-x-1" />
                </Button>
              </Link>
            </CardFooter>
          </Card>
        </motion.div>
      </motion.div>

      

      <motion.div 
        className="mt-8" 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        transition={{ delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-semibold">Recent Activity</h2>
          <Button variant="ghost" size="sm" className="gap-1 text-muted-foreground hover:text-foreground transition-colors">
            View All
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        {/* <Card className="bg-card/50 border-border/50 transition-all duration-300 hover:border-primary/10">
          <div className="px-6 py-8 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted mb-4">
              <Clock className="h-6 w-6 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium mb-2">No recent activity</h3>
            <p className="text-muted-foreground max-w-[500px] mx-auto">
              You haven't used any features yet. Get started by selecting one of the options above.
            </p>
          </div>
        </Card> */}
      </motion.div>
    </div>
  )
}
