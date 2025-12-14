"use client"

import * as React from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"

export interface ToastProps {
  id: string
  title?: string
  description?: string
  variant?: "default" | "destructive" | "success"
  onClose: () => void
}

export function Toast({ id, title, description, variant = "default", onClose }: ToastProps) {
  React.useEffect(() => {
    const timer = setTimeout(() => {
      onClose()
    }, 5000)

    return () => clearTimeout(timer)
  }, [onClose])

  return (
    <div
      className={`fixed top-4 right-4 z-50 w-full max-w-sm rounded-lg border p-4 shadow-lg transition-all ${
        variant === "default" ? "bg-gray-800 border-gray-700 text-white" :
        variant === "destructive" ? "bg-red-900 border-red-700 text-white" :
        "bg-green-900 border-green-700 text-white"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1">
          {title && (
            <div className="font-semibold text-sm text-white">
              {title}
            </div>
          )}
          {description && (
            <div className={`text-sm mt-1 ${
              variant === "default" ? "text-gray-300" :
              variant === "destructive" ? "text-red-200" :
              "text-green-200"
            }`}>
              {description}
            </div>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}