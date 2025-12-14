"use client";

import { useState } from "react";
import Library from "@/components/library";
import { Button } from "@/components/ui/button";
import { Home, Search, Library as LibraryIcon } from "lucide-react";

export default function HomePage() {
  const [activeNav, setActiveNav] = useState<string>("home");

  return (
    <div className="flex flex-col h-screen bg-black text-white p-4">
      {/* Navigation Bar */}
      <ul className="space-y-2">
        {[
          { icon: Home, label: "Home", id: "home" },
          { icon: Search, label: "Search", id: "search" },
          { icon: LibraryIcon, label: "Your Library", id: "library" },
        ].map((item) => (
          <li key={item.id}>
            <Button
              variant="ghost"
              className={`w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-white/10 ${
                activeNav === item.id ? "bg-white/20" : ""
              }`}
              onClick={() => setActiveNav(item.id)}
            >
              <item.icon className="w-5 h-5" />
              {item.label}
            </Button>
          </li>
        ))}
      </ul>

      {/* Conditionally Render Library Section */}
      <div className="mt-6 flex-1 overflow-auto">
        {activeNav === "library" && <Library />}
      </div>
    </div>
  );
}
