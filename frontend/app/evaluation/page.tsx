"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  fetchAllEvaluations,
  fetchBenchmarkCatalog,
  fetchBenchmarkDetail,
  fetchQueryEvaluations,
  deleteQueryEvaluation,
  EvaluationRunResult,
  BenchmarkCatalogItem,
  QueryEvaluation,
} from "@/lib/api";
import QueryEvaluationModal from "@/components/QueryEvaluationModal";
import {
  ArrowLeft,
  RefreshCw,
  Zap,
  Clock,
  CheckCircle2,
  BarChart3,
  Layers,
  FileText,
  Activity,
  Trash2,
  Eye,
  Award,
  Search,
  Database,
} from "lucide-react";

export default function EvaluationPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "benchmarks" | "financebench" | "queries">("overview");

  // State
  const [comparisonData, setComparisonData] = useState<{
    v1_dense: EvaluationRunResult;
    v2_1_bm25: EvaluationRunResult;
    v2_2_hybrid: EvaluationRunResult;
  } | null>(null);

  const [benchmarkCatalog, setBenchmarkCatalog] = useState<BenchmarkCatalogItem[]>([]);
  const [selectedBenchmarkDetail, setSelectedBenchmarkDetail] = useState<any | null>(null);
  const [isBenchmarkModalOpen, setIsBenchmarkModalOpen] = useState(false);

  const [queryEvals, setQueryEvals] = useState<QueryEvaluation[]>([]);
  const [selectedQueryEval, setSelectedQueryEval] = useState<QueryEvaluation | null>(null);
  const [isQueryModalOpen, setIsQueryModalOpen] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [comp, bCatalog, qEvals] = await Promise.all([
        fetchAllEvaluations().catch(() => null),
        fetchBenchmarkCatalog().catch(() => ({ total: 0, benchmarks: [] })),
        fetchQueryEvaluations(50).catch(() => []),
      ]);

      if (comp) setComparisonData(comp);
      if (bCatalog && bCatalog.benchmarks) setBenchmarkCatalog(bCatalog.benchmarks);
      if (qEvals) setQueryEvals(qEvals);
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation center data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleOpenBenchmark = async (bId: string) => {
    try {
      const detail = await fetchBenchmarkDetail(bId);
      setSelectedBenchmarkDetail(detail);
      setIsBenchmarkModalOpen(true);
    } catch (e: any) {
      alert(`Failed to load benchmark detail: ${e.message}`);
    }
  };

  const handleDeleteQueryEval = async (evalId: string) => {
    if (confirm("Are you sure you want to delete this query evaluation record?")) {
      await deleteQueryEvaluation(evalId);
      setQueryEvals((prev) => prev.filter((q) => q.evaluation_id !== evalId));
    }
  };

  const v1 = comparisonData?.v1_dense;
  const v2_1 = comparisonData?.v2_1_bm25;
  const v2_2 = comparisonData?.v2_2_hybrid;

  const filteredQueryEvals = queryEvals.filter(
    (q) =>
      q.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
      q.answer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      q.retrieval_mode.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header Navigation */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                NEXUS RAG Evaluation Center
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  Observability & Benchmarks
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Multi-version retrieval comparison, per-query latency profiling & FinanceBench stress test metrics
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={loadData}
              disabled={loading}
              className="px-3.5 py-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800 text-xs font-semibold flex items-center gap-2 transition-all"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh Data
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800/80 pb-2">
          <button
            onClick={() => setActiveTab("overview")}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === "overview"
                ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <BarChart3 className="w-4 h-4" /> Overview & Models
          </button>
          <button
            onClick={() => setActiveTab("benchmarks")}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === "benchmarks"
                ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <Award className="w-4 h-4" /> Benchmarks ({benchmarkCatalog.length})
          </button>
          <button
            onClick={() => setActiveTab("financebench")}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === "financebench"
                ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <Database className="w-4 h-4" /> FinanceBench Stress Test
          </button>
          <button
            onClick={() => setActiveTab("queries")}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === "queries"
                ? "bg-purple-600 text-white shadow-lg shadow-purple-600/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            }`}
          >
            <Activity className="w-4 h-4" /> Query Evaluations ({queryEvals.length})
          </button>
        </div>

        {/* Tab 1: Overview & Models */}
        {activeTab === "overview" && (
          <div className="space-y-6 animate-fadeIn">
            {/* Version Strategy Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* V1 Card */}
              <div className="glass-panel p-5 rounded-2xl border border-blue-500/20 space-y-3 relative overflow-hidden">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-blue-500/20 text-blue-300 border border-blue-500/30">
                      V1 Baseline
                    </span>
                    <h3 className="text-base font-bold text-white mt-1">V1 — Dense Retrieval</h3>
                  </div>
                  <span className="text-xs px-2 py-1 rounded bg-blue-900/50 text-blue-300 font-semibold border border-blue-700/50">
                    🔒 Frozen Baseline
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Semantic vector embeddings with SentenceTransformers <code className="text-blue-300">all-MiniLM-L6-v2</code> & Qdrant vector database.
                </p>

                {v1 && (
                  <div className="pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-400 block text-[10px]">Recall@1</span>
                      <strong className="text-white text-sm">{(v1.aggregate_recall_at_1 * 100).toFixed(1)}%</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Avg Latency</span>
                      <strong className="text-blue-400 text-sm">{v1.average_retrieval_latency_ms.toFixed(2)} ms</strong>
                    </div>
                  </div>
                )}
              </div>

              {/* V2.1 Card */}
              <div className="glass-panel p-5 rounded-2xl border border-purple-500/20 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-purple-500/20 text-purple-300 border border-purple-500/30">
                      V2.1 Lexical
                    </span>
                    <h3 className="text-base font-bold text-white mt-1">V2.1 — BM25 Retrieval</h3>
                  </div>
                  <span className="text-xs px-2 py-1 rounded bg-purple-900/50 text-purple-300 font-semibold border border-purple-700/50">
                    ⚡ Lexical Engine
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  BM25 okapi keyword matching. Exceptional accuracy for exact entity names, codes, and literal text phrases.
                </p>

                {v2_1 && (
                  <div className="pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-400 block text-[10px]">Recall@1</span>
                      <strong className="text-white text-sm">{(v2_1.aggregate_recall_at_1 * 100).toFixed(1)}%</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Avg Latency</span>
                      <strong className="text-purple-400 text-sm">{v2_1.average_retrieval_latency_ms.toFixed(2)} ms</strong>
                    </div>
                  </div>
                )}
              </div>

              {/* V2.2 Card */}
              <div className="glass-panel p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/10 space-y-3 relative">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      V2.2 Production
                    </span>
                    <h3 className="text-base font-bold text-white mt-1">V2.2 — Hybrid Retrieval</h3>
                  </div>
                  <span className="text-xs px-2 py-1 rounded bg-emerald-900/60 text-emerald-300 font-semibold border border-emerald-700/50">
                    ⭐ Recommended
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Dense + BM25 combined using Reciprocal Rank Fusion (RRF <code className="text-emerald-300">k=60</code>). High precision and recall.
                </p>

                {v2_2 && (
                  <div className="pt-2 border-t border-slate-800 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-slate-400 block text-[10px]">Recall@1</span>
                      <strong className="text-emerald-400 text-sm">{(v2_2.aggregate_recall_at_1 * 100).toFixed(1)}%</strong>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px]">Avg Latency</span>
                      <strong className="text-emerald-400 text-sm">{v2_2.average_retrieval_latency_ms.toFixed(2)} ms</strong>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Matrix Benchmark Comparison */}
            {comparisonData && (
              <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-purple-400" />
                  Official 12-Question Baseline Metric Matrix
                </h3>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950 text-slate-400 uppercase border-b border-slate-800">
                      <tr>
                        <th className="px-4 py-3">Metric</th>
                        <th className="px-4 py-3 text-blue-300">V1 — Dense Baseline</th>
                        <th className="px-4 py-3 text-purple-300">V2.1 — BM25 Lexical</th>
                        <th className="px-4 py-3 text-emerald-300">V2.2 — Hybrid RRF</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                      <tr>
                        <td className="px-4 py-3 font-semibold text-slate-300">Recall@1</td>
                        <td className="px-4 py-3 text-slate-200">{(v1!.aggregate_recall_at_1 * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-slate-200">{(v2_1!.aggregate_recall_at_1 * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 font-bold text-emerald-400">{(v2_2!.aggregate_recall_at_1 * 100).toFixed(1)}%</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-semibold text-slate-300">Recall@5</td>
                        <td className="px-4 py-3 text-slate-200">{(v1!.aggregate_recall_at_5 * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-slate-200">{(v2_1!.aggregate_recall_at_5 * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 font-bold text-emerald-400">{(v2_2!.aggregate_recall_at_5 * 100).toFixed(1)}%</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-semibold text-slate-300">MRR@10</td>
                        <td className="px-4 py-3 text-slate-200">{v1!.aggregate_mrr_at_10.toFixed(4)}</td>
                        <td className="px-4 py-3 text-slate-200">{v2_1!.aggregate_mrr_at_10.toFixed(4)}</td>
                        <td className="px-4 py-3 font-bold text-emerald-400">{v2_2!.aggregate_mrr_at_10.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-semibold text-slate-300">NDCG@10</td>
                        <td className="px-4 py-3 text-slate-200">{v1!.aggregate_ndcg_at_10.toFixed(4)}</td>
                        <td className="px-4 py-3 text-slate-200">{v2_1!.aggregate_ndcg_at_10.toFixed(4)}</td>
                        <td className="px-4 py-3 font-bold text-emerald-400">{v2_2!.aggregate_ndcg_at_10.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <td className="px-4 py-3 font-semibold text-slate-300">Avg Isolated Retrieval Latency</td>
                        <td className="px-4 py-3 text-blue-300 font-medium">{v1!.average_retrieval_latency_ms.toFixed(2)} ms</td>
                        <td className="px-4 py-3 text-purple-300 font-medium">{v2_1!.average_retrieval_latency_ms.toFixed(2)} ms</td>
                        <td className="px-4 py-3 text-emerald-300 font-medium">{v2_2!.average_retrieval_latency_ms.toFixed(2)} ms</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Benchmarks Catalog */}
        {activeTab === "benchmarks" && (
          <div className="space-y-4 animate-fadeIn">
            <h3 className="text-sm font-bold text-white">Registered Benchmark Runs & Test Artifacts</h3>
            <div className="overflow-x-auto glass-panel rounded-2xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Benchmark Name</th>
                    <th className="px-4 py-3">Type</th>
                    <th className="px-4 py-3">Mode</th>
                    <th className="px-4 py-3">Questions</th>
                    <th className="px-4 py-3">Recall@1</th>
                    <th className="px-4 py-3">Recall@5</th>
                    <th className="px-4 py-3">MRR@10</th>
                    <th className="px-4 py-3">Avg Latency</th>
                    <th className="px-4 py-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {benchmarkCatalog.map((b) => (
                    <tr key={b.benchmark_id} className="hover:bg-slate-900/50">
                      <td className="px-4 py-3 font-semibold text-white font-mono">{b.benchmark_id}</td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-800 text-slate-300 border border-slate-700">
                          {b.benchmark_type}
                        </span>
                      </td>
                      <td className="px-4 py-3 uppercase font-semibold text-purple-300">{b.retrieval_mode}</td>
                      <td className="px-4 py-3 text-slate-300">{b.total_questions}</td>
                      <td className="px-4 py-3 font-semibold text-cyan-300">{(b.recall_at_1 * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 font-semibold text-cyan-300">{(b.recall_at_5 * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-slate-300">{b.mrr_at_10.toFixed(4)}</td>
                      <td className="px-4 py-3 text-indigo-300">{b.average_retrieval_latency_ms.toFixed(2)} ms</td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => handleOpenBenchmark(b.benchmark_id)}
                          className="px-2.5 py-1 bg-purple-900/40 hover:bg-purple-800/60 text-purple-300 rounded-lg flex items-center gap-1 text-[11px] font-medium transition-colors"
                        >
                          <Eye className="w-3 h-3" /> Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 3: FinanceBench Stress Test */}
        {activeTab === "financebench" && (
          <div className="space-y-6 animate-fadeIn">
            <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Database className="w-5 h-5 text-emerald-400" />
                    FinanceBench Real-World SEC 10-K Benchmark
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    150 annotated financial questions across 84 SEC 10-K annual reports
                  </p>
                </div>
                <div className="flex gap-2">
                  <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 block">Total Questions</span>
                    <strong className="text-sm font-bold text-white">150</strong>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 block">Required PDFs</span>
                    <strong className="text-sm font-bold text-emerald-400">84 / 84</strong>
                  </div>
                  <div className="bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl text-center">
                    <span className="text-[10px] text-slate-400 block">Missing PDFs</span>
                    <strong className="text-sm font-bold text-cyan-400">0</strong>
                  </div>
                </div>
              </div>

              {/* Status Note */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 space-y-2">
                <div className="font-semibold text-white flex items-center gap-2">
                  <span>ℹ️</span> Single-Document Test 1 Validation Run Completed: <code className="text-purple-300">3M_2018_10K.pdf</code>
                </div>
                <p>
                  FinanceBench evaluation testing is isolated under <code className="text-slate-400">backend/app/evaluation/results/financebench/</code>. Preserves baseline integrity.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: Query Evaluations Feed */}
        {activeTab === "queries" && (
          <div className="space-y-4 animate-fadeIn">
            {/* Search filter */}
            <div className="flex items-center justify-between gap-4">
              <div className="relative flex-1 max-w-md">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                <input
                  type="text"
                  placeholder="Filter query evaluations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-purple-500"
                />
              </div>
              <span className="text-xs text-slate-400">Showing {filteredQueryEvals.length} recorded events</span>
            </div>

            {/* Table */}
            <div className="overflow-x-auto glass-panel rounded-2xl border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-950 text-slate-400 uppercase border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Timestamp</th>
                    <th className="px-4 py-3">Mode</th>
                    <th className="px-4 py-3">Query</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Total Latency</th>
                    <th className="px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {filteredQueryEvals.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                        No query evaluations recorded yet. Ask a query in the Chat UI to capture an execution log!
                      </td>
                    </tr>
                  ) : (
                    filteredQueryEvals.map((q) => (
                      <tr key={q.evaluation_id} className="hover:bg-slate-900/50">
                        <td className="px-4 py-3 font-mono text-slate-400 text-[11px]">
                          {new Date(q.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-800 text-purple-300 border border-slate-700">
                            {q.retrieval_mode}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-medium text-white max-w-xs truncate">{q.query}</td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950 text-emerald-300 border border-emerald-800/50">
                            {q.evaluation_status?.retrieval_status || "detected"}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-semibold text-cyan-400">
                          {q.latency_breakdown?.total_request_ms || 0} ms
                        </td>
                        <td className="px-4 py-3 flex items-center gap-2">
                          <button
                            onClick={() => {
                              setSelectedQueryEval(q);
                              setIsQueryModalOpen(true);
                            }}
                            className="p-1.5 bg-purple-900/40 hover:bg-purple-800/60 text-purple-300 rounded-lg transition-colors"
                            title="Inspect Evaluation"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleDeleteQueryEval(q.evaluation_id)}
                            className="p-1.5 bg-rose-900/30 hover:bg-rose-800/50 text-rose-400 rounded-lg transition-colors"
                            title="Delete Record"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Modal: Query Evaluation Inspector */}
        <QueryEvaluationModal
          evaluation={selectedQueryEval}
          isOpen={isQueryModalOpen}
          onClose={() => setIsQueryModalOpen(false)}
        />

        {/* Modal: Benchmark Detail Inspector */}
        {isBenchmarkModalOpen && selectedBenchmarkDetail && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
              <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950">
                <h3 className="text-base font-bold text-white">
                  Benchmark Run Detail: {selectedBenchmarkDetail.evaluation_version || "Benchmark"}
                </h3>
                <button onClick={() => setIsBenchmarkModalOpen(false)} className="text-slate-400 hover:text-white">
                  ✕
                </button>
              </div>
              <div className="p-6 overflow-y-auto text-xs text-slate-300 space-y-4">
                <pre className="bg-slate-950 p-4 rounded-xl border border-slate-800 overflow-x-auto text-[11px] font-mono text-cyan-300">
                  {JSON.stringify(selectedBenchmarkDetail, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
