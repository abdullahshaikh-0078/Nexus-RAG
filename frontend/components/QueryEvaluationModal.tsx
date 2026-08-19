"use client";

import React from "react";
import { QueryEvaluation } from "@/lib/api";

interface QueryEvaluationModalProps {
  evaluation: QueryEvaluation | null;
  isOpen: boolean;
  onClose: () => void;
}

export default function QueryEvaluationModal({
  evaluation,
  isOpen,
  onClose,
}: QueryEvaluationModalProps) {
  if (!isOpen || !evaluation) return null;

  const lat = evaluation.latency_breakdown || {};
  const status = evaluation.evaluation_status || {};

  const getModeBadge = (mode: string) => {
    switch (mode?.toLowerCase()) {
      case "dense":
        return <span className="px-2 py-0.5 rounded text-xs font-semibold bg-blue-900/60 text-blue-300 border border-blue-700/50">V1 — Dense</span>;
      case "bm25":
        return <span className="px-2 py-0.5 rounded text-xs font-semibold bg-purple-900/60 text-purple-300 border border-purple-700/50">V2.1 — BM25</span>;
      default:
        return <span className="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-900/60 text-emerald-300 border border-emerald-700/50">V2.2 — Hybrid RRF ⭐</span>;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden animate-fadeIn">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-3">
            <span className="text-xl">📊</span>
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Query Evaluation Inspector
                {getModeBadge(evaluation.retrieval_mode)}
              </h2>
              <p className="text-xs text-slate-400 font-mono">ID: {evaluation.evaluation_id}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-300">
          {/* Status Badges */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs">
              <span className="text-slate-400">Context Status:</span>
              <span className={`font-semibold ${status.retrieval_status === "relevant_context_detected" ? "text-emerald-400" : "text-amber-400"}`}>
                {status.retrieval_status === "relevant_context_detected" ? "✅ Relevant Context Detected" : "⚠️ No Relevant Context"}
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs">
              <span className="text-slate-400">Answer Status:</span>
              <span className={`font-semibold ${status.answer_status === "answer_generated" ? "text-cyan-400" : "text-amber-400"}`}>
                {status.answer_status === "answer_generated" ? "⚡ Answer Generated" : "ℹ️ Insufficient Evidence"}
              </span>
            </div>

            <div className="ml-auto text-xs text-slate-400">
              {new Date(evaluation.timestamp).toLocaleString()}
            </div>
          </div>

          {/* User Query */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">User Query</h4>
            <p className="text-white font-medium">{evaluation.query}</p>
          </div>

          {/* Latency Breakdown Bar & Cards */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Step-by-Step Latency Breakdown ({lat.total_request_ms || 0} ms Total)
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-center text-xs">
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">Embedding</div>
                <div className="font-bold text-blue-400">{lat.embedding_ms || 0} ms</div>
              </div>
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">Dense Search</div>
                <div className="font-bold text-indigo-400">{lat.dense_search_ms || 0} ms</div>
              </div>
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">BM25 Search</div>
                <div className="font-bold text-purple-400">{lat.bm25_search_ms || 0} ms</div>
              </div>
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">RRF Fusion</div>
                <div className="font-bold text-emerald-400">{lat.rrf_fusion_ms || 0} ms</div>
              </div>
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">Context Expansion</div>
                <div className="font-bold text-teal-400">{lat.context_expansion_ms || 0} ms</div>
              </div>
              <div className="bg-slate-800/50 p-2.5 rounded-lg border border-slate-700/50">
                <div className="text-slate-400 mb-1">LLM Gen</div>
                <div className="font-bold text-cyan-400">{lat.llm_generation_ms || 0} ms</div>
              </div>
            </div>
          </div>

          {/* Generated Answer */}
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Generated Answer</h4>
            <p className="text-slate-200 whitespace-pre-wrap leading-relaxed">{evaluation.answer}</p>
          </div>

          {/* Retrieved Context Chunks Table */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Retrieved Context Chunks ({evaluation.citations?.length || 0})
            </h4>
            <div className="overflow-x-auto border border-slate-800 rounded-xl">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase border-b border-slate-800">
                  <tr>
                    <th className="px-3 py-2.5">Rank</th>
                    <th className="px-3 py-2.5">Chunk ID</th>
                    <th className="px-3 py-2.5">Score</th>
                    <th className="px-3 py-2.5">Dense Rank</th>
                    <th className="px-3 py-2.5">BM25 Rank</th>
                    <th className="px-3 py-2.5">Content Preview</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 bg-slate-900/50">
                  {(evaluation.citations || []).map((c, i) => (
                    <tr key={i} className="hover:bg-slate-800/30">
                      <td className="px-3 py-2.5 font-bold text-cyan-400">#{i + 1}</td>
                      <td className="px-3 py-2.5 font-mono text-slate-300">{c.chunk_id || `chunk_${c.chunk_index}`}</td>
                      <td className="px-3 py-2.5 font-semibold text-white">
                        {typeof c.score === "number" ? c.score.toFixed(4) : c.score}
                      </td>
                      <td className="px-3 py-2.5 text-slate-400">{c.dense_rank ? `#${c.dense_rank}` : "-"}</td>
                      <td className="px-3 py-2.5 text-slate-400">{c.bm25_rank ? `#${c.bm25_rank}` : "-"}</td>
                      <td className="px-3 py-2.5 text-slate-300 max-w-md truncate">{c.content}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950/50 flex justify-between items-center text-xs text-slate-500">
          <span>Observability Event Recorded via Nexus RAG Evaluation Engine</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
