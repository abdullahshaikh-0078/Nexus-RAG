"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { fetchSystemHealth, SystemHealth } from "@/lib/api";
import { Layers, Activity, CheckCircle2, AlertTriangle, XCircle, PlusCircle, BarChart3, MessageSquare } from "lucide-react";

interface NavbarProps {
  onNewChat?: () => void;
}

export default function Navbar({ onNewChat }: NavbarProps) {
  const pathname = usePathname();
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function checkHealth() {
      try {
        const data = await fetchSystemHealth();
        setHealth(data);
      } catch (err) {
        setHealth(null);
      } finally {
        setLoading(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 25000);
    return () => clearInterval(interval);
  }, []);

  const renderStatusBadge = () => {
    if (loading) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 animate-pulse">
          <Activity className="w-3.5 h-3.5 animate-spin text-blue-400" /> Connecting...
        </span>
      );
    }
    if (!health) {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
          <XCircle className="w-3.5 h-3.5" /> Offline
        </span>
      );
    }
    if (health.status === "healthy") {
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <CheckCircle2 className="w-3.5 h-3.5" /> API Connected
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
        <AlertTriangle className="w-3.5 h-3.5" /> Service Degraded
      </span>
    );
  };

  return (
    <header className="sticky top-0 z-40 w-full glass-panel border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Branding & Nav Links */}
        <div className="flex items-center space-x-6">
          <Link href="/" className="flex items-center space-x-3 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-white">NEXUS RAG</span>
                <span className="px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-blue-400 bg-blue-500/10 border border-blue-500/20 rounded">
                  V1.5
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Enterprise Context Engine</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1 border-l border-slate-800 pl-6">
            <Link
              href="/"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                pathname === "/"
                  ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" /> Workspace Chat
            </Link>
            <Link
              href="/evaluation"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                pathname === "/evaluation"
                  ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" /> Evaluation Suite
            </Link>
          </nav>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          {renderStatusBadge()}

          {onNewChat && (
            <button
              onClick={onNewChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 transition-all shadow-md shadow-blue-600/20"
              title="Reset current conversation state"
            >
              <PlusCircle className="w-3.5 h-3.5" /> New Chat
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
