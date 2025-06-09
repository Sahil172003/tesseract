"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, FileText, ImageIcon, Database, BarChart, LogOut, Settings, User, Menu, Sparkles } from "lucide-react"
import { motion } from "framer-motion"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { ThemeToggle } from "@/components/theme-toggle"

export function AppSidebar() {
  const pathname = usePathname()
  const { isMobile, state } = useSidebar()

  // Define main navigation items
  const mainNavItems = [
    {
      title: "Dashboard",
      href: "/dashboard",
      icon: Home,
    },
   
    
  ]

  // Define document chat sub-items
  const documentChatItems = [
    {
      title: "Chat with Document",
      href: "/dashboard/document-chat/ocr",
      icon: FileText,
    },
    {
      title: "Chat with SQLite",
      href: "/dashboard/sqlite",
      icon: Database,
    },
    {
      title: "Data Visualization",
      href: "/dashboard/visualization",
      icon: BarChart,
    },
    {
      title: "Image Generation",
      href: "/dashboard/image-generation",
      icon: ImageIcon,
    },
  ]

  // Define user menu items
  const userMenuItems = [
    
    {
      title: "Sign Out",
      href: "/sign-in",
      icon: LogOut,
    },
  ]

  const sidebarItemVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: (i: number) => ({
      opacity: 1,
      x: 0,
      transition: {
        delay: i * 0.05,
      },
    }),
  }

  return (
    <>
      {isMobile && (
        <div className="fixed top-4 left-4 z-50">
          <SidebarTrigger asChild>
            <Button variant="outline" size="icon" className="rounded-full shadow-md">
              <Menu className="h-5 w-5" />
            </Button>
          </SidebarTrigger>
        </div>
      )}

      <Sidebar variant="inset">
        <SidebarHeader className="flex h-16 items-center border-b px-4">
          <Link href="/dashboard" className="flex items-center gap-2 font-semibold group">
            <div className="flex items-center justify-center w-8 h-8 rounded-md bg-primary/10 transition-all duration-300 group-hover:bg-primary/20">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <span className="gradient-text text-xl transition-all group-hover:scale-105">GenAI App</span>
          </Link>
        </SidebarHeader>
        <SidebarContent>
          <div className="p-1 mt-2"></div>
          <SidebarMenu>
            {mainNavItems.map((item, i) => (
              <motion.div key={item.href} custom={i} initial="hidden" animate="visible" variants={sidebarItemVariants}>
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.href || pathname.startsWith(item.href + "/")}
                    tooltip={item.title}
                    className="h-12 text-base transition-all duration-200 hover:translate-x-1"
                  >
                    <Link href={item.href}>
                      <item.icon className="h-5 w-5" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </motion.div>
            ))}
          </SidebarMenu>

          
            <>
              <SidebarSeparator className="my-3" />
              <div className="px-4 py-2">
                <h4 className="text-xs font-medium text-sidebar-foreground/70 tracking-wider uppercase">Start chating with</h4>
              </div>
              <SidebarMenu>
                {documentChatItems.map((item, i) => (
                  <motion.div
                    key={item.href}
                    custom={i}
                    initial="hidden"
                    animate="visible"
                    variants={sidebarItemVariants}
                  >
                    <SidebarMenuItem>
                      <SidebarMenuButton
                        asChild
                        isActive={pathname === item.href}
                        tooltip={item.title}
                        className="h-10 text-sm transition-all duration-200 hover:translate-x-1"
                      >
                        <Link href={item.href}>
                          <item.icon className="h-4 w-4" />
                          <span>{item.title}</span>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  </motion.div>
                ))}
              </SidebarMenu>
            </>
          
        </SidebarContent>
        <SidebarFooter>
          <SidebarSeparator className="my-2" />
          
          <div className="p-2 flex items-center justify-between">
            <div className="text-xs font-medium text-sidebar-foreground/70 px-2">Theme</div>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div><ThemeToggle /></div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Toggle theme</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          
          <div className="p-4">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="ghost" 
                  className="w-full justify-start px-2 h-auto border border-transparent transition-all hover:bg-sidebar-accent/20 hover:border-sidebar-accent/20 rounded-lg"
                >
                  <div className="flex items-center gap-3 py-1.5">
                    <Avatar className="h-9 w-9 border-2 border-primary/20 transition-all hover:border-primary">
                      <AvatarImage src="/placeholder-user.jpg" alt="User" />
                      <AvatarFallback>
  <User className="w-4 h-4" />
</AvatarFallback>
                    </Avatar>
                    <div className="flex flex-col items-start text-left">
                      <span className="text-sm font-medium">My Account</span>
                      <span className="text-xs text-sidebar-foreground/70"></span>
                    </div>
                  </div>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {userMenuItems.map((item) => (
                  <DropdownMenuItem key={item.href} asChild>
                    <Link href={item.href} className="flex items-center gap-2 cursor-pointer">
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </SidebarFooter>
      </Sidebar>
    </>
  )
}
