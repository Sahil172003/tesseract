import type React from "react"
export const metadata = {
  title: "GenAI Application",
  description: "A comprehensive GenAI application with document chat and image generation",
    generator: 'v0.dev'
}

import ClientLayout from "./client-layout"

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <ClientLayout>{children}</ClientLayout>
}


import './globals.css'