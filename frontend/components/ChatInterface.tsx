"use client";

import { useState, useRef, useEffect } from "react";
import { queryRag, ChatQueryResponse, SourceCitation, DocumentMetadata, QueryEvaluation, fetchQueryEvaluationDetail } from "@/lib/api";
import SourceCitations from "./SourceCitations";
import SourceInspector from "./SourceInspector";
import QueryEvaluationModal from "./QueryEvaluationModal";
import {
  Send,
  Bot,
  User,
  Sparkles,
  Clock,
  Cpu,
  Loader2,
  AlertTriangle,
  CornerDownLeft,
  Layers,
  Info,
  HelpCircle,
  BarChart2,
} from "lucide-react";

export interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  responseMeta?: ChatQueryResponse;
}

interface ChatInterfaceProps {
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export default function ChatInterface({
  documents,
  selectedDocId,
  messages,
  setMessages,
}: ChatInterfaceProps) {
  const [inputQuery, setInputQuery] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<"hybrid" | "dense" | "bm25">("hybrid");
  const [showInfo, setShowInfo] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<SourceCitation | null>(null);
  
  // Per-Query Evaluation Inspector Modal State
  const [selectedEval, setSelectedEval] = useState<QueryEvaluation | null>(null);
  const [isEvalModalOpen, setIsEvalModalOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = async () => {
    const query = inputQuery.trim();
    if (!query || loading) return;

    setError(null);
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      sender: "user",
      text: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setLoading(true);

    try {
      const docIdsFilter = selectedDocId ? [selectedDocId] : undefined;
      const res = await queryRag(query, 4, docIdsFilter, retrievalMode);

      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        sender: "assistant",
        text: res.answer,
        responseMeta: res,
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMessage = err.message || "Failed to process RAG query.";
      setError(errorMessage);
      const errorMsg: Message = {
        id: `err-${Date.now()}`,
        sender: "assistant",
        text: `Error: ${errorMessage}`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleViewEvaluation = async (evalId?: string, responseMeta?: ChatQueryResponse) => {
    if (!evalId) {
      if (responseMeta) {
        // Fallback construct transient QueryEvaluation object
        const tempEval: QueryEvaluation = {
          evaluation_id: `transient-${Date.now()}`,
          timestamp: new Date().toISOString(),
          query: responseMeta.query,
          retrieval_mode: responseMeta.retrieval_mode,
          citations: responseMeta.sources,
          final_context: responseMeta.sources,
          answer: responseMeta.answer,
          latency_breakdown: responseMeta.latency_breakdown || { total_request_ms: responseMeta.processing_time_seconds * 1000 },
          evaluation_status: {
            retrieval_status: responseMeta.sources.length ? "relevant_context_detected" : "no_relevant_context",
            answer_status: "answer_generated",
          },
        };
        setSelectedEval(tempEval);
        setIsEvalModalOpen(true);
      }
      return;
    }

    try {
      const evalDetail = await fetchQueryEvaluationDetail(evalId);
      setSelectedEval(evalDetail);
      setIsEvalModalOpen(true);
    } catch (e) {
      if (responseMeta) {
        const tempEval: QueryEvaluation = {
          evaluation_id: evalId,
          timestamp: new Date().toISOString(),
          query: responseMeta.query,
          retrieval_mode: responseMeta.retrieval_mode,
          citations: responseMeta.sources,
          final_context: responseMeta.sources,
          answer: responseMeta.answer,
          latency_breakdown: responseMeta.latency_breakdown || { total_request_ms: responseMeta.processing_time_seconds * 1000 },
          evaluation_status: {
            retrieval_status: responseMeta.sources.length ? "relevant_context_detected" : "no_relevant_context",
            answer_status: "answer_generated",
          },
        };
        setSelectedEval(tempEval);
        setIsEvalModalOpen(true);
      }
    }
  };

  const activeDoc = documents.find((d) => d.document_id === selectedDocId);

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950/40 p-4 relative overflow-hidden">
      {/* Header bar */}
      <div className="pb-3 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              NEXUS Chat Assistant
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {retrievalMode.toUpperCase()}
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">Ask questions over active ingested knowledge base</p>
          </div>
        </div>

        {/* Retrieval Mode Selector */}
        <div className="flex items-center gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => setRetrievalMode("hybrid")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              retrievalMode === "hybrid"
                ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-md shadow-emerald-900/40"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            V2.2 — Hybrid ⭐
          </button>
          <button
            type="button"
            onClick={() => setRetrievalMode("bm25")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              retrievalMode === "bm25"
                ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md shadow-purple-900/40"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            V2.1 — BM25
          </button>
          <button
            type="button"
            onClick={() => setRetrievalMode("dense")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              retrievalMode === "dense"
                ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-md shadow-blue-900/40"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            V1 — Dense
          </button>
          <button
            type="button"
            onClick={() => setShowInfo(!showInfo)}
            className="p-1 text-slate-400 hover:text-white transition-colors"
            title="Retrieval Mode Information"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Info Tooltip Banner */}
      {showInfo && (
        <div className="mt-3 p-3.5 rounded-xl bg-slate-900/95 border border-slate-700/80 text-xs text-slate-300 shadow-xl space-y-2 animate-fadeIn">
          <div className="font-semibold text-white flex items-center gap-1.5">
            <Info className="w-4 h-4 text-purple-400" />
            NEXUS RAG Retrieval Mode Explanations:
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 pt-1 text-[11px]">
            <div className="p-2 rounded-lg bg-blue-950/40 border border-blue-800/40">
              <strong className="text-blue-300 block mb-0.5">V1 — Dense (Frozen Baseline):</strong>
              Semantic vector search with SentenceTransformers all-MiniLM-L6-v2. Strong for conceptual queries.
            </div>
            <div className="p-2 rounded-lg bg-purple-950/40 border border-purple-800/40">
              <strong className="text-purple-300 block mb-0.5">V2.1 — BM25 Lexical:</strong>
              BM25 okapi keyword matching. Strong for exact entity names, acronyms, and technical codes.
            </div>
            <div className="p-2 rounded-lg bg-emerald-950/40 border border-emerald-800/40">
              <strong className="text-emerald-300 block mb-0.5">V2.2 — Hybrid (Recommended):</strong>
              Combines Dense & BM25 via Reciprocal Rank Fusion (RRF). Standard production retrieval engine.
            </div>
          </div>
        </div>
      )}

      {/* Scope Filter Status */}
      {selectedDocId && activeDoc && (
        <div className="mt-3 bg-blue-600/10 border border-blue-500/20 px-3 py-1.5 rounded-xl flex items-center justify-between text-xs text-blue-300">
          <span className="flex items-center gap-1.5 font-medium truncate">
            <Sparkles className="w-3.5 h-3.5 text-blue-400 shrink-0" />
            Query scoped to: <strong className="text-white">{activeDoc.filename}</strong>
          </span>
          <span className="text-[10px] text-blue-400 uppercase font-bold bg-blue-500/20 px-1.5 py-0.5 rounded">
            Filtered
          </span>
        </div>
      )}

      {/* Main Conversation Feed */}
      <div className="flex-1 overflow-y-auto space-y-5 my-3 pr-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
            <div className="w-12 h-12 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
              <Bot className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200">Start a RAG Conversation</h3>
            <p className="text-xs text-slate-400 max-w-md">
              Ask questions about your uploaded documents. <strong>V2.2 Hybrid Search (RRF)</strong> is active by default to fuse vector semantics and BM25 keywords.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.sender === "assistant" && (
                <div className="w-8 h-8 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-purple-400 shrink-0 mt-0.5">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div className="max-w-[85%] sm:max-w-[78%] space-y-2">
                <div
                  className={`p-4 rounded-2xl text-xs leading-relaxed ${
                    msg.sender === "user"
                      ? "bg-blue-600 text-white rounded-tr-none font-medium shadow-lg shadow-blue-600/10"
                      : "glass-panel text-slate-200 rounded-tl-none border-slate-800/90"
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.text}</div>

                  {/* Metadata, Sources, and View Evaluation button */}
                  {msg.responseMeta && (
                    <>
                      <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex flex-wrap items-center justify-between gap-3 text-[10px] text-slate-400">
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="flex items-center gap-1 font-semibold text-purple-300 uppercase">
                            <Layers className="w-3 h-3 text-purple-400" /> Mode: {msg.responseMeta.retrieval_mode}
                          </span>
                          <span className="flex items-center gap-1">
                            <Cpu className="w-3 h-3 text-blue-400" /> {msg.responseMeta.llm_provider} ({msg.responseMeta.model_name})
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-indigo-400" /> {msg.responseMeta.processing_time_seconds}s
                          </span>
                        </div>

                        {/* View Evaluation Control Button */}
                        <button
                          onClick={() => handleViewEvaluation(msg.responseMeta?.evaluation_id, msg.responseMeta)}
                          className="px-2.5 py-1 bg-purple-900/40 hover:bg-purple-800/60 text-purple-300 hover:text-purple-100 border border-purple-700/60 rounded-lg flex items-center gap-1.5 text-[11px] font-medium transition-colors"
                        >
                          <BarChart2 className="w-3.5 h-3.5 text-purple-400" />
                          View Evaluation
                        </button>
                      </div>

                      <SourceCitations
                        sources={msg.responseMeta.sources}
                        onSelectCitation={(cite) => setSelectedCitation(cite)}
                      />
                    </>
                  )}
                </div>
              </div>

              {msg.sender === "user" && (
                <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-md shadow-blue-600/20">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))
        )}

        {/* Loading Indicator */}
        {loading && (
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-purple-400 shrink-0">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="glass-panel p-3.5 rounded-2xl text-xs text-purple-300 flex items-center gap-2 border border-slate-800">
              <Sparkles className="w-4 h-4 animate-pulse text-purple-400" />
              Retrieving context via {retrievalMode.toUpperCase()} search & synthesizing response...
            </div>
          </div>
        )}

        {/* Error Display Card */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="pt-2 border-t border-slate-800/80">
        <div className="relative flex items-center">
          <textarea
            ref={textareaRef}
            rows={1}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask NEXUS a question using ${retrievalMode.toUpperCase()} search...`}
            className="w-full bg-slate-900/90 text-xs text-white placeholder-slate-500 rounded-xl pl-4 pr-12 py-3 border border-slate-800 focus:outline-none focus:border-purple-500/60 focus:ring-1 focus:ring-purple-500/50 resize-none transition-all"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!inputQuery.trim() || loading}
            className="absolute right-2.5 p-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-500 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-purple-600/20"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Source Citation Inspector Modal */}
      {selectedCitation && (
        <SourceInspector
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
        />
      )}

      {/* Query Evaluation Inspector Modal */}
      <QueryEvaluationModal
        evaluation={selectedEval}
        isOpen={isEvalModalOpen}
        onClose={() => setIsEvalModalOpen(false)}
      />
    </div>
  );
}
