const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export interface ServiceHealth {
  status: "healthy" | "degraded" | "unhealthy";
  details?: string;
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "unhealthy";
  app_name: string;
  environment: string;
  services: {
    mongodb: ServiceHealth;
    qdrant: ServiceHealth;
    embeddings: ServiceHealth;
    llm: ServiceHealth;
  };
  timestamp: string;
}

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  char_count: number;
  chunk_count: number;
  upload_timestamp: string;
  status: "processed" | "processing" | "failed";
  error_message?: string;
}

export interface SourceCitation {
  document_id: string;
  document_name: string;
  chunk_id: string;
  chunk_index: number;
  score: number;
  content: string;
  dense_rank?: number;
  bm25_rank?: number;
  rrf_score?: number;
}

export interface LatencyBreakdown {
  embedding_ms?: number;
  dense_search_ms?: number;
  bm25_search_ms?: number;
  rrf_fusion_ms?: number;
  context_expansion_ms?: number;
  llm_generation_ms?: number;
  total_request_ms?: number;
}

export interface ChatQueryResponse {
  query: string;
  answer: string;
  sources: SourceCitation[];
  retrieval_mode: string;
  llm_provider: string;
  model_name: string;
  processing_time_seconds: number;
  evaluation_id?: string;
  latency_breakdown?: LatencyBreakdown;
}

export interface QueryEvaluation {
  evaluation_id: string;
  timestamp: string;
  query: string;
  document_ids?: string[];
  retrieval_mode: string;
  dense_results?: any[];
  bm25_results?: any[];
  hybrid_results?: any[];
  final_context: SourceCitation[];
  answer: string;
  citations: SourceCitation[];
  latency_breakdown: LatencyBreakdown;
  evaluation_status: {
    retrieval_status: string;
    answer_status: string;
  };
  ground_truth?: any;
  retrieval_metrics?: any;
  answer_metrics?: any;
}

export interface BenchmarkCatalogItem {
  benchmark_id: string;
  file_name: string;
  relative_path: string;
  benchmark_type: string;
  dataset_version: string;
  evaluation_version: string;
  retrieval_mode: string;
  timestamp: string;
  total_questions: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr_at_10: number;
  ndcg_at_10: number;
  average_retrieval_latency_ms: number;
}

export interface QuestionEvalResult {
  question_id: string;
  question: string;
  category: string;
  first_relevant_rank?: number;
  dense_rank?: number;
  bm25_rank?: number;
  hybrid_rank?: number;
  rrf_score?: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr_at_10: number;
  ndcg_at_10: number;
  retrieval_latency_ms: number;
  retrieved_chunk_ids: string[];
  retrieved_snippets: string[];
}

export interface EvaluationRunResult {
  dataset_version?: string;
  evaluation_version: string;
  retrieval_mode: string;
  bm25?: boolean;
  fusion_method?: string;
  rrf_k?: number;
  timestamp: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  retrieval_top_k: number;
  total_questions: number;
  aggregate_recall_at_1: number;
  aggregate_recall_at_3: number;
  aggregate_recall_at_5: number;
  aggregate_recall_at_10: number;
  aggregate_mrr_at_10: number;
  aggregate_ndcg_at_10: number;
  average_retrieval_latency_ms: number;
  question_results: QuestionEvalResult[];
}

export async function fetchSystemHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status}): ${res.statusText}`);
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<DocumentMetadata> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Document upload failed.");
  }

  const data = await res.json();
  return data.document;
}

export async function listDocuments(): Promise<DocumentMetadata[]> {
  const res = await fetch(`${API_BASE_URL}/documents/`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch documents list (${res.status}): ${res.statusText}`);
  }
  const data = await res.json();
  return data.documents || [];
}

export async function deleteDocument(documentId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Failed to delete document ${documentId}`);
  }
  return true;
}

export async function queryRag(
  query: string,
  topK: number = 4,
  documentIds?: string[],
  retrievalMode: string = "hybrid"
): Promise<ChatQueryResponse> {
  const validDocIds =
    documentIds && documentIds.length > 0
      ? documentIds.filter((id) => id && id.trim() !== "" && id.trim() !== "string")
      : undefined;

  const res = await fetch(`${API_BASE_URL}/chat/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      top_k: topK,
      document_ids: validDocIds,
      retrieval_mode: retrievalMode,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to process RAG query.");
  }

  return res.json();
}

export async function fetchLatestEvaluation(mode: string = "hybrid"): Promise<EvaluationRunResult> {
  const res = await fetch(`${API_BASE_URL}/evaluation/results?mode=${mode}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to fetch evaluation results.");
  }
  return res.json();
}

export async function fetchAllEvaluations(): Promise<{
  v1_dense: EvaluationRunResult;
  v2_1_bm25: EvaluationRunResult;
  v2_2_hybrid: EvaluationRunResult;
}> {
  const res = await fetch(`${API_BASE_URL}/evaluation/results/all`, {
    cache: "no-store",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to fetch all evaluation results.");
  }
  return res.json();
}

export async function triggerEvaluationRun(mode: string = "hybrid"): Promise<EvaluationRunResult> {
  const res = await fetch(`${API_BASE_URL}/evaluation/run?mode=${mode}`, {
    method: "POST",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || "Failed to execute evaluation suite.");
  }
  return res.json();
}

export async function fetchBenchmarkCatalog(): Promise<{ total: number; benchmarks: BenchmarkCatalogItem[] }> {
  const res = await fetch(`${API_BASE_URL}/evaluation/benchmarks`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch benchmark catalog.");
  return res.json();
}

export async function fetchBenchmarkDetail(benchmarkId: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/evaluation/benchmarks/${benchmarkId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch benchmark detail for ${benchmarkId}`);
  return res.json();
}

export async function fetchQueryEvaluations(limit: number = 50): Promise<QueryEvaluation[]> {
  const res = await fetch(`${API_BASE_URL}/evaluation/query-evaluations?limit=${limit}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch query evaluations.");
  return res.json();
}

export async function fetchQueryEvaluationDetail(evalId: string): Promise<QueryEvaluation> {
  const res = await fetch(`${API_BASE_URL}/evaluation/query-evaluations/${evalId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch query evaluation ${evalId}`);
  return res.json();
}

export async function deleteQueryEvaluation(evalId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE_URL}/evaluation/query-evaluations/${evalId}`, { method: "DELETE" });
  return res.ok;
}
