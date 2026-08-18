"use client";

import { useEffect, useState } from "react";
import Navbar from "@/components/Navbar";
import {
  fetchAllEvaluations,
  triggerEvaluationRun,
  EvaluationRunResult,
} from "@/lib/api";
import {
  BarChart3,
  Clock,
  ShieldCheck,
  Cpu,
  Info,
  Play,
  Loader2,
  AlertCircle,
  FileCode,
  Hash,
  ArrowRight,
  TrendingUp,
  TrendingDown,
} from "lucide-react";

export default function EvaluationPage() {
  const [evalData, setEvalData] = useState<{
    v1_dense: EvaluationRunResult | null;
    v2_1_bm25: EvaluationRunResult | null;
  }>({ v1_dense: null, v2_1_bm25: null });
  const [loading, setLoading] = useState(true);
  const [runningMode, setRunningMode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAllEvaluations();
      setEvalData(data);
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation results.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
  }, []);

  const handleRunEvaluation = async (mode: "dense" | "bm25") => {
    setRunningMode(mode);
    setError(null);
    try {
      const res = await triggerEvaluationRun(mode);
      if (mode === "bm25") {
        setEvalData((prev) => ({ ...prev, v2_1_bm25: res }));
      } else {
        setEvalData((prev) => ({ ...prev, v1_dense: res }));
      }
    } catch (err: any) {
      setError(err.message || `Failed to execute ${mode} evaluation.`);
    } finally {
      setRunningMode(null);
    }
  };

  const v1 = evalData.v1_dense;
  const v2_1 = evalData.v2_1_bm25;

  const renderDelta = (v1Val: number, v2Val: number, isPercentage: boolean = true) => {
    const diff = v2Val - v1Val;
    if (Math.abs(diff) < 0.0001) {
      return <span className="text-slate-500 font-mono text-[10px]">0.0</span>;
    }
    const formatted = isPercentage
      ? `${(diff * 100).toFixed(1)}%`
      : diff.toFixed(4);
    if (diff > 0) {
      return (
        <span className="text-emerald-400 font-bold flex items-center gap-0.5 text-[10px]">
          <TrendingUp className="w-3 h-3" /> +{formatted}
        </span>
      );
    }
    return (
      <span className="text-rose-400 font-bold flex items-center gap-0.5 text-[10px]">
        <TrendingDown className="w-3 h-3" /> {formatted}
      </span>
    );
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
                <BarChart3 className="w-6 h-6 text-blue-400" /> Evaluation & Benchmarking Dashboard
              </h1>
              <span className="px-2.5 py-0.5 text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-full">
                V1 Dense vs V2.1 BM25 Comparison
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Scientific side-by-side evaluation measuring Recall@K, MRR@10, NDCG@10, and isolated retrieval latency.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleRunEvaluation("dense")}
              disabled={!!runningMode || loading}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-blue-600/20 hover:bg-blue-600 text-blue-300 hover:text-white border border-blue-500/30 transition-all cursor-pointer disabled:opacity-50"
            >
              {runningMode === "dense" ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              Run V1 Dense Eval
            </button>

            <button
              onClick={() => handleRunEvaluation("bm25")}
              disabled={!!runningMode || loading}
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 transition-all cursor-pointer disabled:opacity-50"
            >
              {runningMode === "bm25" ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              Run V2.1 BM25 Eval
            </button>
          </div>
        </div>

        {/* Feedback / Error Banners */}
        {error && (
          <div className="flex items-center gap-2 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div className="py-16 text-center space-y-2">
            <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto" />
            <p className="text-xs font-semibold text-slate-300">Loading evaluation benchmarks...</p>
          </div>
        ) : !v1 && !v2_1 ? (
          <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center space-y-3">
            <Info className="w-8 h-8 text-slate-500 mx-auto" />
            <h3 className="text-sm font-semibold text-slate-200">No evaluation runs available</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Click &quot;Run V1 Dense Eval&quot; or &quot;Run V2.1 BM25 Eval&quot; above to execute benchmarking suites.
            </p>
          </div>
        ) : (
          <>
            {/* System Parameter Specifications */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
              <div className="flex items-center space-x-4 flex-wrap gap-y-1">
                <span className="flex items-center gap-1.5 font-semibold text-slate-200">
                  <Cpu className="w-4 h-4 text-blue-400" /> Model: {v1?.embedding_model || v2_1?.embedding_model || "all-MiniLM-L6-v2"}
                </span>
                <span className="text-slate-600">|</span>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <FileCode className="w-4 h-4 text-indigo-400" /> Chunk Size: {v1?.chunk_size || 1000} (Overlap: {v1?.chunk_overlap || 150})
                </span>
                <span className="text-slate-600">|</span>
                <span className="flex items-center gap-1.5 text-slate-400">
                  <Hash className="w-4 h-4 text-cyan-400" /> Top-K: {v1?.retrieval_top_k || 10}
                </span>
              </div>
              <div className="text-[11px] text-slate-400">
                Dataset: <strong className="text-slate-200">Attention Is All You Need (12 Questions)</strong>
              </div>
            </div>

            {/* Side-by-Side Metric Comparison Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Recall@1 */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400">Recall @ 1</span>
                  {v1 && v2_1 && renderDelta(v1.aggregate_recall_at_1, v2_1.aggregate_recall_at_1)}
                </div>
                <div className="flex items-baseline justify-between pt-1 border-t border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-blue-400 block font-semibold">V1 Dense</span>
                    <span className="text-lg font-bold text-blue-300">
                      {v1 ? `${(v1.aggregate_recall_at_1 * 100).toFixed(1)}%` : "--"}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-indigo-400 block font-semibold">V2.1 BM25</span>
                    <span className="text-lg font-bold text-indigo-300">
                      {v2_1 ? `${(v2_1.aggregate_recall_at_1 * 100).toFixed(1)}%` : "--"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Recall@5 */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400">Recall @ 5</span>
                  {v1 && v2_1 && renderDelta(v1.aggregate_recall_at_5, v2_1.aggregate_recall_at_5)}
                </div>
                <div className="flex items-baseline justify-between pt-1 border-t border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-blue-400 block font-semibold">V1 Dense</span>
                    <span className="text-lg font-bold text-blue-300">
                      {v1 ? `${(v1.aggregate_recall_at_5 * 100).toFixed(1)}%` : "--"}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-indigo-400 block font-semibold">V2.1 BM25</span>
                    <span className="text-lg font-bold text-indigo-300">
                      {v2_1 ? `${(v2_1.aggregate_recall_at_5 * 100).toFixed(1)}%` : "--"}
                    </span>
                  </div>
                </div>
              </div>

              {/* MRR@10 */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400">MRR @ 10</span>
                  {v1 && v2_1 && renderDelta(v1.aggregate_mrr_at_10, v2_1.aggregate_mrr_at_10, false)}
                </div>
                <div className="flex items-baseline justify-between pt-1 border-t border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-blue-400 block font-semibold">V1 Dense</span>
                    <span className="text-lg font-bold text-blue-300">
                      {v1 ? v1.aggregate_mrr_at_10.toFixed(4) : "--"}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-indigo-400 block font-semibold">V2.1 BM25</span>
                    <span className="text-lg font-bold text-indigo-300">
                      {v2_1 ? v2_1.aggregate_mrr_at_10.toFixed(4) : "--"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Avg Latency */}
              <div className="glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400 flex items-center gap-1">
                    <Clock className="w-3 h-3 text-amber-400" /> Avg Latency
                  </span>
                  {v1 && v2_1 && (
                    <span className="text-[10px] text-slate-400 font-mono">
                      {(v2_1.average_retrieval_latency_ms - v1.average_retrieval_latency_ms).toFixed(1)} ms
                    </span>
                  )}
                </div>
                <div className="flex items-baseline justify-between pt-1 border-t border-slate-800/80">
                  <div>
                    <span className="text-[10px] text-blue-400 block font-semibold">V1 Dense</span>
                    <span className="text-lg font-bold text-amber-400">
                      {v1 ? `${v1.average_retrieval_latency_ms.toFixed(1)} ms` : "--"}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-indigo-400 block font-semibold">V2.1 BM25</span>
                    <span className="text-lg font-bold text-amber-300">
                      {v2_1 ? `${v2_1.average_retrieval_latency_ms.toFixed(1)} ms` : "--"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Per-Question Comparison Table */}
            <div className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-blue-400" /> Side-by-Side Question Breakdown (V1 Dense vs V2.1 BM25)
                </h3>
                <span className="text-xs text-slate-400 font-medium">
                  12 Test Questions Evaluated
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
                      <th className="py-3 px-2">ID</th>
                      <th className="py-3 px-2">Category</th>
                      <th className="py-3 px-2 min-w-[240px]">Question</th>
                      <th className="py-3 px-2 text-center">V1 Rank</th>
                      <th className="py-3 px-2 text-center">V2.1 Rank</th>
                      <th className="py-3 px-2 text-center">V1 R@5</th>
                      <th className="py-3 px-2 text-center">V2.1 R@5</th>
                      <th className="py-3 px-2 text-center">V1 MRR</th>
                      <th className="py-3 px-2 text-center">V2.1 MRR</th>
                      <th className="py-3 px-2 text-right">V1 ms</th>
                      <th className="py-3 px-2 text-right">V2.1 ms</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-200">
                    {(v1?.question_results || v2_1?.question_results || []).map((q, idx) => {
                      const q1 = v1?.question_results[idx];
                      const q2 = v2_1?.question_results[idx];
                      return (
                        <tr key={q.question_id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="py-3 px-2 font-mono font-bold text-blue-400">{q.question_id}</td>
                          <td className="py-3 px-2">
                            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-indigo-300 border border-slate-700">
                              {q.category}
                            </span>
                          </td>
                          <td className="py-3 px-2 font-medium text-slate-200">{q.question}</td>

                          {/* V1 Rank */}
                          <td className="py-3 px-2 text-center font-mono">
                            {q1?.first_relevant_rank ? (
                              <span className="text-emerald-400 font-bold">#{q1.first_relevant_rank}</span>
                            ) : (
                              <span className="text-rose-400 font-bold">--</span>
                            )}
                          </td>

                          {/* V2.1 Rank */}
                          <td className="py-3 px-2 text-center font-mono">
                            {q2?.first_relevant_rank ? (
                              <span className="text-indigo-400 font-bold">#{q2.first_relevant_rank}</span>
                            ) : (
                              <span className="text-rose-400 font-bold">--</span>
                            )}
                          </td>

                          {/* V1 Recall@5 */}
                          <td className="py-3 px-2 text-center font-mono">
                            {q1 && q1.recall_at_5 > 0 ? (
                              <span className="text-emerald-400 font-semibold">1.0</span>
                            ) : (
                              <span className="text-slate-500">0.0</span>
                            )}
                          </td>

                          {/* V2.1 Recall@5 */}
                          <td className="py-3 px-2 text-center font-mono">
                            {q2 && q2.recall_at_5 > 0 ? (
                              <span className="text-indigo-400 font-semibold">1.0</span>
                            ) : (
                              <span className="text-slate-500">0.0</span>
                            )}
                          </td>

                          {/* MRR */}
                          <td className="py-3 px-2 text-center font-mono text-blue-300">
                            {q1 ? q1.mrr_at_10.toFixed(2) : "--"}
                          </td>
                          <td className="py-3 px-2 text-center font-mono text-indigo-300">
                            {q2 ? q2.mrr_at_10.toFixed(2) : "--"}
                          </td>

                          {/* Latency */}
                          <td className="py-3 px-2 text-right font-mono text-slate-400">
                            {q1 ? `${q1.retrieval_latency_ms.toFixed(1)}ms` : "--"}
                          </td>
                          <td className="py-3 px-2 text-right font-mono text-indigo-300">
                            {q2 ? `${q2.retrieval_latency_ms.toFixed(1)}ms` : "--"}
                          </td>
                        </tr>
                      );
                    })}
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
