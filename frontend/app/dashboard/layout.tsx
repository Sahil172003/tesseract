"use client"

import type React from "react"
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarInset } from "@/components/ui/sidebar"
import { motion } from "framer-motion"
import { Container } from "@/components/ui/container"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar />
      <SidebarInset className="transition-all duration-300">
        <motion.div
          className="flex-1"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="h-full pb-8">
            <Container size="xl" className="h-full pt-6 md:pt-8 lg:pt-10">
              {children}
            </Container>
          </div>
        </motion.div>
      </SidebarInset>
    </div>
  )
}
