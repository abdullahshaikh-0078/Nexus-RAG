"use client";

import { useEffect, useState } from "react";
import Navbar from "@/components/Navbar";
import {
  fetchLatestEvaluation,
  triggerEvaluationRun,
  EvaluationRunResult,
} from "@/lib/api";
import {
  BarChart3,
  Target,
  Clock,
  ShieldCheck,
  Cpu,
  Info,
  Play,
  Layers,
  Loader2,
  CheckCircle2,
  AlertCircle,
  FileCode,
  Hash,
} from "lucide-react";

export default function EvaluationPage() {
  const [evalResult, setEvalResult] = useState<EvaluationRunResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchLatestEvaluation();
      setEvalResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation results.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
  }, []);

  const handleRunEvaluation = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await triggerEvaluationRun();
      setEvalResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to execute evaluation suite.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                <BarChart3 className="w-6 h-6 text-blue-400" /> Evaluation & Benchmarking Engine
              </h1>
              <span className="px-2.5 py-0.5 text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                {evalResult?.evaluation_version || "V1 Dense Retrieval Baseline"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Scientific evaluation framework measuring Recall@K, MRR@10, NDCG@10, and isolated retrieval latency.
            </p>
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={running || loading}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white shadow-lg shadow-blue-600/20 transition-all cursor-pointer"
          >
            {running ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Running Evaluation Suite...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" /> Run Evaluation Suite
              </>
            )}
          </button>
        </div>

        {/* Feedback / Error Banners */}
        {error && (
          <div className="flex items-center gap-2 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* System Baseline Specs Bar */}
        {evalResult && (
          <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex items-center space-x-4">
              <span className="flex items-center gap-1.5 font-semibold text-slate-200">
                <Cpu className="w-4 h-4 text-blue-400" /> Model: {evalResult.embedding_model}
              </span>
              <span className="text-slate-600">|</span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <FileCode className="w-4 h-4 text-indigo-400" /> Chunk Size: {evalResult.chunk_size} (Overlap: {evalResult.chunk_overlap})
              </span>
              <span className="text-slate-600">|</span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <Hash className="w-4 h-4 text-cyan-400" /> Top-K: {evalResult.retrieval_top_k}
              </span>
            </div>
            <div className="text-[11px] text-slate-400">
              Evaluated on <strong className="text-slate-200">{evalResult.total_questions} Test Questions</strong> • {new Date(evalResult.timestamp).toLocaleString()}
            </div>
          </div>
        )}

        {/* Aggregate Metric Cards */}
        {loading ? (
          <div className="py-12 text-center space-y-2">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto" />
            <p className="text-xs font-semibold text-slate-300">Loading evaluation baseline results...</p>
          </div>
        ) : !evalResult ? (
          <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center space-y-3">
            <Info className="w-8 h-8 text-slate-500 mx-auto" />
            <h3 className="text-sm font-semibold text-slate-200">No evaluation run available</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Click &quot;Run Evaluation Suite&quot; above to execute the V1 Dense Retrieval baseline benchmarks.
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-3">
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">Recall @ 1</span>
                <div className="text-xl font-bold text-blue-400">
                  {(evalResult.aggregate_recall_at_1 * 100).toFixed(1)}%
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">Recall @ 3</span>
                <div className="text-xl font-bold text-blue-400">
                  {(evalResult.aggregate_recall_at_3 * 100).toFixed(1)}%
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">Recall @ 5</span>
                <div className="text-xl font-bold text-indigo-400">
                  {(evalResult.aggregate_recall_at_5 * 100).toFixed(1)}%
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">Recall @ 10</span>
                <div className="text-xl font-bold text-indigo-400">
                  {(evalResult.aggregate_recall_at_10 * 100).toFixed(1)}%
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">MRR @ 10</span>
                <div className="text-xl font-bold text-emerald-400">
                  {evalResult.aggregate_mrr_at_10.toFixed(4)}
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">NDCG @ 10</span>
                <div className="text-xl font-bold text-cyan-400">
                  {evalResult.aggregate_ndcg_at_10.toFixed(4)}
                </div>
              </div>

              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-1.5 col-span-2 sm:col-span-1">
                <span className="text-[11px] font-semibold text-slate-400 flex items-center gap-1">
                  <Clock className="w-3 h-3 text-amber-400" /> Avg Latency
                </span>
                <div className="text-xl font-bold text-amber-400">
                  {evalResult.average_retrieval_latency_ms.toFixed(1)} ms
                </div>
              </div>
            </div>

            {/* Per-Question Breakdown Table */}
            <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-blue-400" /> Per-Question Evaluation Breakdown
                </h3>
                <span className="text-xs text-slate-400 font-medium">
                  {evalResult.question_results.length} Test Questions Evaluated
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
                      <th className="py-3 px-2">ID</th>
                      <th className="py-3 px-2">Category</th>
                      <th className="py-3 px-2 min-w-[260px]">Question</th>
                      <th className="py-3 px-2 text-center">First Rank</th>
                      <th className="py-3 px-2 text-center">Recall@5</th>
                      <th className="py-3 px-2 text-center">Recall@10</th>
                      <th className="py-3 px-2 text-center">MRR</th>
                      <th className="py-3 px-2 text-center">NDCG</th>
                      <th className="py-3 px-2 text-right">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-200">
                    {evalResult.question_results.map((q) => (
                      <tr key={q.question_id} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 px-2 font-mono font-bold text-blue-400">{q.question_id}</td>
                        <td className="py-3 px-2">
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-indigo-300 border border-slate-700">
                            {q.category}
                          </span>
                        </td>
                        <td className="py-3 px-2 font-medium text-slate-200">{q.question}</td>
                        <td className="py-3 px-2 text-center font-mono">
                          {q.first_relevant_rank ? (
                            <span className="text-emerald-400 font-bold">#{q.first_relevant_rank}</span>
                          ) : (
                            <span className="text-rose-400 font-bold">--</span>
                          )}
                        </td>
                        <td className="py-3 px-2 text-center">
                          {q.recall_at_5 > 0 ? (
                            <span className="text-emerald-400 font-semibold">1.0</span>
                          ) : (
                            <span className="text-slate-500">0.0</span>
                          )}
                        </td>
                        <td className="py-3 px-2 text-center">
                          {q.recall_at_10 > 0 ? (
                            <span className="text-emerald-400 font-semibold">1.0</span>
                          ) : (
                            <span className="text-slate-500">0.0</span>
                          )}
                        </td>
                        <td className="py-3 px-2 text-center font-mono text-indigo-300">
                          {q.mrr_at_10.toFixed(3)}
                        </td>
                        <td className="py-3 px-2 text-center font-mono text-cyan-300">
                          {q.ndcg_at_10.toFixed(3)}
                        </td>
                        <td className="py-3 px-2 text-right font-mono text-amber-300">
                          {q.retrieval_latency_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
