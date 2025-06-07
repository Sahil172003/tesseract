import { cn } from "@/lib/utils"
import * as React from "react"

interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  as?: React.ElementType
  size?: "default" | "small" | "large" | "xl" | "full"
}

const Container = React.forwardRef<HTMLDivElement, ContainerProps>(
  ({ className, as: Component = "div", size = "default", ...props }, ref) => {
    return (
      <Component
        ref={ref}
        className={cn(
          "mx-auto w-full px-4",
          size === "small" && "max-w-3xl",
          size === "default" && "max-w-6xl",
          size === "large" && "max-w-7xl",
          size === "xl" && "max-w-[1400px]",
          size === "full" && "max-w-full",
          "sm:px-6 lg:px-8",
          className
        )}
        {...props}
      />
    )
  }
)

Container.displayName = "Container"

export { Container }