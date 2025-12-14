"use client"

import * as React from "react"
import { Toast, ToastProps } from "@/components/ui/toast"

type ToastInput = Omit<ToastProps, "id" | "onClose">

export function useToast() {
  const [toasts, setToasts] = React.useState<ToastProps[]>([])

  const toast = React.useCallback((input: ToastInput) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newToast: ToastProps = {
      ...input,
      id,
      onClose: () => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      },
    }
    setToasts((prev) => [...prev, newToast])
  }, [])

  const ToastContainer = React.useCallback(() => (
    <>
      {toasts.map((toast) => (
        <Toast key={toast.id} {...toast} />
      ))}
    </>
  ), [toasts])

  return { toast, ToastContainer }
}