import type * as React from "react"

import { cn } from "@/lib/utils"

export const ChartContainer = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return <div className={cn("rounded-md border bg-card text-card-foreground p-4", className)} {...props} />
}
ChartContainer.displayName = "ChartContainer"

export const Chart = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return <div className={cn("h-full w-full", className)} {...props} />
}
Chart.displayName = "Chart"

export const ChartTooltip = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return (
    <div className={cn("rounded-md border bg-popover text-popover-foreground p-4 shadow-md", className)} {...props} />
  )
}
ChartTooltip.displayName = "ChartTooltip"

export const ChartTooltipContent = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return <div className={cn("text-sm", className)} {...props} />
}
ChartTooltipContent.displayName = "ChartTooltipContent"

export const ChartLegend = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return <div className={cn("flex items-center space-x-2", className)} {...props} />
}
ChartLegend.displayName = "ChartLegend"

export const ChartLegendItem = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  return <div className={cn("flex items-center space-x-1 text-sm", className)} {...props} />
}
ChartLegendItem.displayName = "ChartLegendItem"
