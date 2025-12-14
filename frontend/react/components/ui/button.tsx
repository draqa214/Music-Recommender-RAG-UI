import { cn } from "@/lib/utils"
import { ButtonHTMLAttributes, forwardRef } from "react"

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost"
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "default", className, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "px-4 py-2 rounded-md font-medium transition",
          variant === "ghost" ? "bg-transparent hover:bg-white/10" : "bg-white text-black",
          className
        )}
        {...props}
      />
    )
  }
)

Button.displayName = "Button"
export { Button }
