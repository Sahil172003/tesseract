"use client"

import { useState } from "react"
import { ImageIcon, Sparkles, Download, Loader2, Settings, RefreshCw, Wand2 } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Badge } from "@/components/ui/badge"
import { imageGenerationApi } from "@/lib/api/image-generation"

export default function ImageGenerationPage() {
  const { toast } = useToast()
  const [prompt, setPrompt] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImages, setGeneratedImages] = useState<string[]>([])
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [resolution, setResolution] = useState("1024x1024")
  const [style, setStyle] = useState("realistic")
  const [numImages, setNumImages] = useState(1) // Changed from 4 to 1
  const [promptSuggestions, setPromptSuggestions] = useState<string[]>([
    "A futuristic cityscape with flying cars and neon lights",
    "A serene mountain landscape with a lake at sunset",
    "A magical forest with glowing plants and mystical creatures",
    "An underwater scene with colorful coral reefs and exotic fish",
    "A steampunk-inspired mechanical dragon",
  ])

  // Generate images - FIXED VERSION
  const handleGenerateImages = async () => {
    if (!prompt.trim()) {
      toast({
        title: "Prompt required",
        description: "Please enter a prompt to generate images.",
        variant: "destructive",
      })
      return
    }

    setIsGenerating(true)
    setGeneratedImages([])
    setSelectedImage(null)

    try {
      const response = await imageGenerationApi.generateImages({
        prompt: prompt.trim(),
        n: numImages,
        size: resolution // Use the actual resolution setting
      })

      setGeneratedImages(response.images)
      setSelectedImage(response.images[0])
      
      toast({
        title: "Images generated",
        description: `Generated ${response.images.length} images based on your prompt.`,
      })
    } catch (error) {
      console.error("Image generation error:", error)
      toast({
        title: "Generation failed",
        description: "Failed to generate images. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
    }
  }

  // Handle image selection
  const handleImageSelect = (image: string) => {
    setSelectedImage(image)
  }

  // Handle image download
  const handleDownloadImage = () => {
    if (!selectedImage) return

    // Create download link for base64 image
    const link = document.createElement('a')
    link.href = selectedImage
    link.download = `generated-image-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    toast({
      title: "Image downloaded",
      description: "The image has been downloaded to your device.",
    })
  }

  // Handle regenerate
  const handleRegenerate = () => {
    handleGenerateImages()
  }

  // Handle prompt suggestion click
  const handlePromptSuggestion = (suggestion: string) => {
    setPrompt(suggestion)
  }

  // Animation variants
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
    hidden: { opacity: 0, scale: 0.9 },
    show: { opacity: 1, scale: 1 },
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <h1 className="text-3xl font-bold tracking-tight gradient-text">Image Generation</h1>
        <p className="text-muted-foreground mt-1">Generate stunning images using AI with text prompts.</p>
      </motion.div>

      <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Card className="overflow-hidden">
            <CardContent className="p-6">
              <div className="space-y-4">
                <div>
                  <Textarea
                    placeholder="Describe the image you want to generate..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="min-h-[100px] resize-none rounded-xl border-muted-foreground/20 focus:border-primary text-base"
                  />
                </div>

                <div className="flex flex-wrap gap-2">
                  {promptSuggestions.map((suggestion, index) => (
                    <Badge
                      key={index}
                      variant="outline"
                      className="cursor-pointer hover:bg-primary/10 transition-colors"
                      onClick={() => handlePromptSuggestion(suggestion)}
                    >
                      {suggestion}
                    </Badge>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    onClick={handleGenerateImages}
                    disabled={isGenerating || !prompt.trim()}
                    className="flex-1 h-12 gap-2 shadow-lg hover:shadow-primary/25 transition-all duration-300"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Wand2 className="h-5 w-5" />
                        Generate
                      </>
                    )}
                  </Button>

                  <Sheet>
                    <SheetTrigger asChild>
                      <Button variant="outline" className="h-12 gap-2">
                        <Settings className="h-5 w-5" />
                        Settings
                      </Button>
                    </SheetTrigger>
                    <SheetContent className="sm:max-w-md">
                      <SheetHeader>
                        <SheetTitle>Generation Settings</SheetTitle>
                        <SheetDescription>Customize your image generation settings.</SheetDescription>
                      </SheetHeader>
                      <div className="space-y-6 py-6">
                        <div className="space-y-2">
                          <label className="text-sm font-medium">Resolution</label>
                          <Select value={resolution} onValueChange={setResolution}>
                            <SelectTrigger className="h-12 rounded-lg">
                              <SelectValue placeholder="Select resolution" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="512x512">512x512</SelectItem>
                              <SelectItem value="1024x1024">1024x1024</SelectItem>
                              <SelectItem value="1024x1792">1024x1792 (Portrait)</SelectItem>
                              <SelectItem value="1792x1024">1792x1024 (Landscape)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-medium">Style</label>
                          <Select value={style} onValueChange={setStyle}>
                            <SelectTrigger className="h-12 rounded-lg">
                              <SelectValue placeholder="Select style" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="realistic">Realistic</SelectItem>
                              <SelectItem value="anime">Anime</SelectItem>
                              <SelectItem value="3d">3D Render</SelectItem>
                              <SelectItem value="cartoon">Cartoon</SelectItem>
                              <SelectItem value="painting">Painting</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <label className="text-sm font-medium">Number of Images</label>
                            <span className="text-sm text-muted-foreground">{numImages}</span>
                          </div>
                          <Slider
                            value={[numImages]}
                            min={1}
                            max={4}
                            step={1}
                            onValueChange={(value) => setNumImages(value[0])}
                            className="py-2"
                          />
                        </div>
                      </div>
                    </SheetContent>
                  </Sheet>
                </div>
              </div>
            </CardContent>
          </Card>

          {isGenerating ? (
            <motion.div
              className="grid grid-cols-2 gap-4 md:grid-cols-4"
              variants={container}
              initial="hidden"
              animate="show"
            >
              {Array.from({ length: numImages }).map((_, i) => (
                <motion.div
                  key={i}
                  className="aspect-square rounded-xl bg-gradient-to-br from-primary/10 to-primary/5 animate-pulse shadow-md"
                  variants={item}
                >
                  <div className="flex h-full items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary/50" />
                  </div>
                </motion.div>
              ))}
            </motion.div>
          ) : generatedImages.length > 0 ? (
            <motion.div
              className="grid grid-cols-2 gap-4 md:grid-cols-4"
              variants={container}
              initial="hidden"
              animate="show"
            >
              <AnimatePresence>
                {generatedImages.map((image, i) => (
                  <motion.div
                    key={i}
                    className={`group relative aspect-square cursor-pointer overflow-hidden rounded-xl border shadow-md transition-all duration-300 hover:shadow-lg ${
                      selectedImage === image ? "ring-2 ring-primary" : ""
                    }`}
                    onClick={() => handleImageSelect(image)}
                    variants={item}
                    whileHover={{ scale: 1.03 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  >
                    <img
                      src={image || "/placeholder.svg"}
                      alt={`Generated image ${i + 1}`}
                      className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                    <div className="absolute bottom-0 left-0 right-0 p-3 text-white opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                      <p className="text-xs font-medium">Image {i + 1}</p>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </motion.div>
          ) : null}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {selectedImage ? (
            <div className="space-y-4 sticky top-6">
              <motion.div
                className="overflow-hidden rounded-xl border shadow-md"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Selected generated image"
                  className="h-full w-full object-cover"
                />
              </motion.div>

              <div className="flex gap-2">
                <Button variant="outline" className="flex-1 gap-2 h-12" onClick={handleDownloadImage}>
                  <Download className="h-4 w-4" />
                  Download
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 gap-2 h-12"
                  onClick={handleRegenerate}
                  disabled={isGenerating}
                >
                  <RefreshCw className="h-4 w-4" />
                  Regenerate
                </Button>
              </div>

              <Card className="overflow-hidden">
                <CardContent className="p-4 space-y-3">
                  <h3 className="font-medium">Image Details</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Resolution:</span>
                      <span className="font-medium">{resolution}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Style:</span>
                      <span className="font-medium capitalize">{style}</span>
                    </div>
                    <div className="pt-2">
                      <span className="text-muted-foreground">Prompt:</span>
                      <p className="mt-1 text-xs bg-muted/50 p-2 rounded-md">{prompt}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <motion.div
              className="flex h-full flex-col items-center justify-center rounded-xl border border-dashed p-8 text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="feature-icon mb-4">
                <ImageIcon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-medium">No image selected</h3>
              <p className="mt-2 text-sm text-muted-foreground">Generate images and select one to view details.</p>
              <Button
                className="mt-4 gap-2 shadow-md hover:shadow-lg transition-all duration-300"
                onClick={() => document.querySelector("textarea")?.focus()}
                disabled={isGenerating}
              >
                <Sparkles className="h-4 w-4" />
                Start Creating
              </Button>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}