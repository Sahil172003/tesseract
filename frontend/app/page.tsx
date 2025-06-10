"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles, FileText, ImageIcon, Zap, Brain, Lightbulb } from "lucide-react"
import { motion } from "framer-motion"
import { Container } from "@/components/ui/container"

export default function Home() {
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
    <div className="flex app flex-col">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <Container size="xl" className="flex h-16 items-center justify-between">
          <div className="flex items-center">
            <Link href="/" className="flex items-center space-x-2 transition-all duration-200 hover:opacity-90">
              <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 transition-transform duration-300 hover:scale-105">
                <Sparkles className="h-5 w-5 text-primary" />
              </div>
              <span className="font-bold gradient-text text-xl">Tesseract"</span>
            </Link>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <nav className="flex items-center space-x-2 sm:space-x-3">
              <Link href="/sign-in" passHref>
                <Button variant="ghost" className="px-4 transition-all duration-200 hover:bg-muted">
                  Sign In
                </Button>
              </Link>
              <Link href="/register" passHref>
                <Button className="px-4 shadow-lg hover:shadow-primary/25 transition-all duration-300">Register</Button>
              </Link>
            </nav>
          </div>
        </Container>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 overflow-hidden">
          <Container size="xl">
            <motion.div
              className="grid gap-8 md:gap-12 lg:gap-16 lg:grid-cols-2 items-center"
              variants={container}
              initial="hidden"
              animate="show"
            >
              <motion.div className="space-y-6" variants={item}>
                <div className="inline-block rounded-full bg-primary/10 px-3 py-1 text-sm text-primary">
                  Introducing Tesseract
                </div>
                <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl">
                  <span className="gradient-text">Unlock the Power</span> of AI
                </h1>
                <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                  Our Tesseract application provides powerful tools for document chat and image generation. Sign up now to
                  explore the possibilities.
                </p>
                <div className="flex flex-col gap-2 min-[400px]:flex-row">
                  <Link href="/register" passHref>
                    <Button className="gap-1 px-6 py-6 text-lg shadow-lg hover:shadow-primary/25 transition-all duration-300 hover:translate-y-[-2px]">
                      Get Started
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  </Link>
                  <Link href="/sign-in" passHref>
                    <Button variant="outline" className="px-6 py-6 text-lg hover:bg-muted/50 transition-all duration-300">
                      Sign In
                    </Button>
                  </Link>
                </div>
              </motion.div>
              <motion.div
                className="relative"
                variants={item}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="absolute -inset-0.5 rounded-xl bg-gradient-to-r from-[hsl(var(--gradient-start))] to-[hsl(var(--gradient-end))] opacity-75 blur-xl"></div>
                <div className="relative grid grid-cols-2 gap-4 sm:gap-6 rounded-xl bg-background p-4 sm:p-6">
                  <div className="highlight-card flex flex-col items-center gap-2 p-6 transition-all duration-300 hover:translate-y-[-2px]">
                    <div className="feature-icon p-2 rounded-lg bg-primary/10 transition-all duration-300">
                      <FileText className="h-7 w-7" />
                    </div>
                    <h3 className="text-xl font-semibold mt-3">Document Chat</h3>
                    <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                      Chat with documents, databases, and visualize data
                    </p>
                  </div>
                  <div className="highlight-card flex flex-col items-center gap-2 p-6 transition-all duration-300 hover:translate-y-[-2px]">
                    <div className="feature-icon p-2 rounded-lg bg-primary/10 transition-all duration-300">
                      <ImageIcon className="h-7 w-7" />
                    </div>
                    <h3 className="text-xl font-semibold mt-3">Image Generation</h3>
                    <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                      Generate stunning images with AI
                    </p>
                  </div>
                  <div className="highlight-card flex flex-col items-center gap-2 p-6 transition-all duration-300 hover:translate-y-[-2px]">
                    <div className="feature-icon p-2 rounded-lg bg-primary/10 transition-all duration-300">
                      <Brain className="h-7 w-7" />
                    </div>
                    <h3 className="text-xl font-semibold mt-3">Smart Analysis</h3>
                    <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                      Extract insights from your data
                    </p>
                  </div>
                  <div className="highlight-card flex flex-col items-center gap-2 p-6 transition-all duration-300 hover:translate-y-[-2px]">
                    <div className="feature-icon p-2 rounded-lg bg-primary/10 transition-all duration-300">
                      <Zap className="h-7 w-7" />
                    </div>
                    <h3 className="text-xl font-semibold mt-3">Fast Results</h3>
                    <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                      Get answers and images in seconds
                    </p>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          </Container>
        </section>

        <section className="w-full py-12 md:py-24 bg-muted/50">
          <Container size="xl">
            <motion.div
              className="flex flex-col items-center justify-center space-y-4 text-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <div className="inline-block rounded-full bg-primary/10 px-3 py-1 text-sm text-primary">How It Works</div>
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Powerful AI Features</h2>
              <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                Our Tesseract application combines multiple AI capabilities to help you work smarter, not harder.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 mt-12">
              <motion.div
                className="highlight-card group transition-all duration-300 hover:translate-y-[-4px]"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                viewport={{ once: true }}
              >
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-all duration-300">
                  <FileText className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-bold">Document Chat</h3>
                <p className="mt-2 text-gray-500 dark:text-gray-400">
                  Upload any document and chat with its contents. Ask questions and get instant answers based on the
                  document.
                </p>
              </motion.div>

              <motion.div
                className="highlight-card group transition-all duration-300 hover:translate-y-[-4px]"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                viewport={{ once: true }}
              >
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-all duration-300">
                  <ImageIcon className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-bold">Image Generation</h3>
                <p className="mt-2 text-gray-500 dark:text-gray-400">
                  Create stunning images from text descriptions. Customize styles, resolutions, and more.
                </p>
              </motion.div>

              <motion.div
                className="highlight-card group transition-all duration-300 hover:translate-y-[-4px]"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                viewport={{ once: true }}
              >
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 group-hover:bg-primary/20 transition-all duration-300">
                  <Lightbulb className="h-7 w-7 text-primary" />
                </div>
                <h3 className="text-xl font-bold">Data Insights</h3>
                <p className="mt-2 text-gray-500 dark:text-gray-400">
                  Visualize your data and extract meaningful insights. Turn complex data into clear visualizations.
                </p>
              </motion.div>
            </div>
          </Container>
        </section>


      </main>
      <footer className="border-t py-8">
        <Container size="xl">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              <p className="text-sm font-medium">
                &copy; {new Date().getFullYear()} Tesseract". All rights reserved.
              </p>
            </div>
            <div className="flex items-center gap-6">
              <Link
                href="/terms"
                className="text-sm text-gray-500 hover:text-gray-900 transition-colors dark:text-gray-400 dark:hover:text-gray-50"
              >
                Terms
              </Link>
              <Link
                href="/privacy"
                className="text-sm text-gray-500 hover:text-gray-900 transition-colors dark:text-gray-400 dark:hover:text-gray-50"
              >
                Privacy
              </Link>
            </div>
          </div>
        </Container>
      </footer>
    </div>
  )
}
