"use client";

import { SourceCitation } from "@/lib/api";
import { X, FileText, Sparkles, Hash, BarChart3 } from "lucide-react";

interface SourceInspectorProps {
  citation: SourceCitation | null;
  onClose: () => void;
}

export default function SourceInspector({ citation, onClose }: SourceInspectorProps) {
  if (!citation) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex justify-end transition-opacity">
      <div className="w-full max-w-lg bg-[#0f172a] border-l border-slate-800 h-full p-6 flex flex-col justify-between shadow-2xl animate-in slide-in-from-right duration-200">
        <div className="space-y-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-5 h-5 text-blue-400" />
              <h3 className="text-base font-semibold text-white">Retrieved Context Citation</h3>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Document metadata cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="glass-panel p-3 rounded-xl">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
                <FileText className="w-3.5 h-3.5 text-blue-400" /> Source File
              </div>
              <p className="text-xs font-semibold text-white truncate">{citation.document_name}</p>
            </div>
            <div className="glass-panel p-3 rounded-xl">
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
                <BarChart3 className="w-3.5 h-3.5 text-emerald-400" /> Similarity Score
              </div>
              <p className="text-xs font-semibold text-emerald-400">
                {(citation.score * 100).toFixed(1)}% ({citation.score})
              </p>
            </div>
          </div>

          <div className="glass-panel p-3 rounded-xl flex items-center justify-between text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-indigo-400" /> Chunk ID: {citation.chunk_id}
            </span>
            <span className="bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded text-[10px] font-medium border border-indigo-500/20">
              Index #{citation.chunk_index}
            </span>
          </div>

          {/* Chunk Content */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Exact Chunk Payload
            </label>
            <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-200 leading-relaxed max-h-[380px] overflow-y-auto whitespace-pre-wrap font-mono">
              {citation.content}
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
}
