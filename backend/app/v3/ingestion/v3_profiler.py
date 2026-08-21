import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class V3Profiler:
    """
    High-resolution timing and metrics collector for V3 Document Ingestion Pipeline.
    Instruments operations using time.perf_counter().
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics: Dict[str, Any] = {
            "source_pdf_loading_s": 0.0,
            "pymupdf_structural_parsing_s": 0.0,
            "v3_doc_ir_construction_s": 0.0,
            "v3_policy_selection_s": 0.0,
            "v3_chunk_generation_s": 0.0,
            "embedding_generation_s": 0.0,
            "qdrant_indexing_s": 0.0,
            "bm25_indexing_s": 0.0,
            "mongodb_updates_s": 0.0,
            "validation_total_s": 0.0,
            "validation_bm25_s": 0.0,
            "validation_dense_s": 0.0,
            "validation_hybrid_s": 0.0,
            "content_hashing_s": 0.0,
            # Embedding specific metrics
            "embedding_model_name": "all-MiniLM-L6-v2",
            "embedding_device": "CPU",
            "embedding_batch_size": 100,
            "embedding_batch_count": 0,
            "total_chunks_embedded": 0,
            # Qdrant specific metrics
            "qdrant_upsert_ops": 0,
            "qdrant_points_per_upsert": 100,
            # BM25 specific metrics
            "bm25_rebuild_ops": 0,
            # Execution totals
            "total_conversion_s": 0.0,
            "chunks_generated": 0,
        }

    def print_report(self) -> str:
        m = self.metrics
        total_s = m["total_conversion_s"] or 1e-6
        chunks = m["chunks_generated"] or 1

        components = [
            ("1. Source PDF loading", m["source_pdf_loading_s"]),
            ("2. PyMuPDF structural parsing", m["pymupdf_structural_parsing_s"]),
            ("3. V3DocumentIR construction", m["v3_doc_ir_construction_s"]),
            ("4. V3 policy/strategy selection", m["v3_policy_selection_s"]),
            ("5. V3 chunk generation", m["v3_chunk_generation_s"]),
            ("6. Embedding generation", m["embedding_generation_s"]),
            ("7. Qdrant indexing/upserts", m["qdrant_indexing_s"]),
            ("8. BM25 indexing", m["bm25_indexing_s"]),
            ("9. MongoDB representation updates", m["mongodb_updates_s"]),
            ("10. 12-point validation", m["validation_total_s"]),
            ("11. Content hashing / Overhead", m["content_hashing_s"]),
        ]

        lines = []
        lines.append("==========================================================================================")
        lines.append("                      V3 INGESTION PERFORMANCE PROFILING REPORT                           ")
        lines.append("==========================================================================================")
        lines.append(f"{'Component':<35} | {'Time (s)':<12} | {'Percentage':<12} | {'Throughput'}")
        lines.append("-" * 90)

        for name, duration in components:
            pct = (duration / total_s) * 100.0
            throughput = f"{chunks / duration:.2f} chunks/sec" if duration > 0 else "N/A"
            lines.append(f"{name:<35} | {duration:<12.4f} | {pct:<11.2f}% | {throughput}")

        lines.append("-" * 90)
        lines.append(f"{'TOTAL V3 CONVERSION':<35} | {total_s:<12.4f} | {'100.00%':<12} | {chunks / total_s:.2f} chunks/sec")
        lines.append("==========================================================================================\n")

        lines.append("--- DETAILED EMBEDDING REPORT ---")
        lines.append(f"• Embedding Model: {m['embedding_model_name']}")
        lines.append(f"• CPU vs GPU Device: {m['embedding_device']}")
        lines.append(f"• Outer Batch Size: {m['embedding_batch_size']}")
        lines.append(f"• Number of Batches: {m['embedding_batch_count']}")
        lines.append(f"• Total Chunks Embedded: {m['total_chunks_embedded']}")
        lines.append(f"• Total Embedding Time: {m['embedding_generation_s']:.4f} s")
        avg_batch = (m['embedding_generation_s'] / m['embedding_batch_count']) if m['embedding_batch_count'] > 0 else 0
        lines.append(f"• Average Time per Batch: {avg_batch:.4f} s")
        lines.append(f"• Average Embedding Throughput: {m['total_chunks_embedded'] / max(m['embedding_generation_s'], 1e-6):.2f} chunks/sec\n")

        lines.append("--- DETAILED QDRANT REPORT ---")
        lines.append(f"• Total Upsert Operations: {m['qdrant_upsert_ops']}")
        lines.append(f"• Points per Upsert: {m['qdrant_points_per_upsert']}")
        lines.append(f"• Total Qdrant Indexing Time: {m['qdrant_indexing_s']:.4f} s\n")

        lines.append("--- DETAILED BM25 REPORT ---")
        lines.append(f"• Total BM25 Indexing Time: {m['bm25_indexing_s']:.4f} s")
        lines.append(f"• Number of Corpus Rebuild Operations: {m['bm25_rebuild_ops']}\n")

        lines.append("--- DETAILED VALIDATION REPORT ---")
        lines.append(f"• Total Validation Time: {m['validation_total_s']:.4f} s")
        lines.append(f"  - Stage 8a (BM25 Search Test): {m['validation_bm25_s']:.4f} s")
        lines.append(f"  - Stage 8b (Dense Vector Search Test): {m['validation_dense_s']:.4f} s")
        lines.append(f"  - Stage 8c (Hybrid RRF Search Test): {m['validation_hybrid_s']:.4f} s\n")

        report_str = "\n".join(lines)
        logger.info(f"\n{report_str}")
        return report_str


v3_profiler = V3Profiler()
