"use client";

import { useState } from "react";
import { SourceCitation } from "@/lib/api";
import { Sparkles, ChevronDown, ChevronUp, FileText, ExternalLink, BarChart2 } from "lucide-react";

interface SourceCitationsProps {
  sources: SourceCitation[];
  onSelectCitation: (citation: SourceCitation) => void;
}

export default function SourceCitations({ sources, onSelectCitation }: SourceCitationsProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-2">
      {/* Toggle header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full text-[11px] font-semibold text-slate-400 hover:text-slate-200 transition-colors"
      >
        <span className="flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          Retrieved Context Sources ({sources.length})
        </span>
        <span className="flex items-center gap-1 text-[10px] text-blue-400">
          {expanded ? "Hide Citations" : "Show Citations"}
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </span>
      </button>

      {/* Expandable Content Cards */}
      {expanded && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 animate-in fade-in duration-200">
          {sources.map((cite, idx) => (
            <div
              key={`${cite.chunk_id}-${idx}`}
              onClick={() => onSelectCitation(cite)}
              className="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-blue-500/50 cursor-pointer transition-all group"
            >
              <div className="flex items-center justify-between gap-1">
                <span className="text-[11px] font-semibold text-slate-200 truncate flex items-center gap-1">
                  <FileText className="w-3 h-3 text-blue-400 shrink-0" />
                  {cite.document_name}
                </span>
                <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 shrink-0">
                  {(cite.score * 100).toFixed(1)}% match
                </span>
              </div>

              <p className="text-[10px] text-slate-400 line-clamp-2 mt-1.5 font-mono leading-relaxed group-hover:text-slate-300">
                {cite.content}
              </p>

              <div className="mt-2 pt-1.5 border-t border-slate-800/60 flex items-center justify-between text-[9px] text-slate-500">
                <span className="flex items-center gap-1">
                  <BarChart2 className="w-2.5 h-2.5 text-indigo-400" /> Chunk #{cite.chunk_index}
                </span>
                <span className="text-blue-400 group-hover:underline flex items-center gap-0.5">
                  Inspect <ExternalLink className="w-2.5 h-2.5" />
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
