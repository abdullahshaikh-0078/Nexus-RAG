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

import { ChatDocument } from "@/lib/api";

export interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  responseMeta?: ChatQueryResponse;
}

interface ChatInterfaceProps {
  activeChatId: string | null;
  chatDocuments: ChatDocument[];
  version: "v1" | "v2.1" | "v2.2" | "v3";
  setVersion: (version: "v1" | "v2.1" | "v2.2" | "v3") => void;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export default function ChatInterface({
  activeChatId,
  chatDocuments,
  version,
  setVersion,
  messages,
  setMessages,
}: ChatInterfaceProps) {
  const [inputQuery, setInputQuery] = useState("");
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
      const activeDocId = chatDocuments.length > 0 ? chatDocuments[0].document_id : undefined;
      const docIdsFilter = activeDocId ? [activeDocId] : undefined;
      const legacyMode = version === "v1" ? "dense" : (version === "v2.1" ? "bm25" : "hybrid");
      const res = await queryRag(query, 4, docIdsFilter, legacyMode, version, "auto", activeChatId || undefined);

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
      setError("No Evaluation ID associated with this query response.");
      return;
    }

    try {
      const detail = await fetchQueryEvaluationDetail(evalId);
      setSelectedEval(detail);
      setIsEvalModalOpen(true);
    } catch (err: any) {
      if (responseMeta) {
        const fallbackEval: QueryEvaluation = {
          evaluation_id: evalId,
          timestamp: new Date().toISOString(),
          query: responseMeta.query,
          retrieval_mode: responseMeta.retrieval_mode,
          final_context: responseMeta.sources,
          answer: responseMeta.answer,
          citations: responseMeta.sources,
          latency_breakdown: responseMeta.latency_breakdown || {},
          evaluation_status: {
            retrieval_status: "SUCCESS",
            answer_status: "GROUNDED",
          },
        };
        setSelectedEval(fallbackEval);
        setIsEvalModalOpen(true);
      } else {
        setError(`Could not fetch query evaluation details: ${err.message}`);
      }
    }
  };

  const activeDoc = chatDocuments[0];

  return (
    <div className="flex flex-col h-full bg-slate-950 p-4 rounded-3xl border border-slate-800 shadow-2xl relative">
      {/* Header Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              NEXUS Chat Assistant
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-purple-500/20 text-purple-300 border border-purple-500/30">
                {version.toUpperCase()}
              </span>
            </h2>
            <p className="text-[11px] text-slate-400">Ask questions over active ingested knowledge base</p>
          </div>
        </div>

        {/* System Version & Backend Strategy Info */}
        <div className="flex flex-wrap items-center gap-2 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800">
          <label className="text-[11px] text-slate-400 font-medium px-1">Pipeline Version:</label>
          <select
            value={version}
            onChange={(e) => {
              setVersion(e.target.value as any);
              setSelectedCitation(null);
            }}
            className="bg-slate-950 text-xs text-white border border-slate-700/80 rounded-lg px-2.5 py-1 focus:outline-none focus:border-purple-500 font-medium"
          >
            <option value="v2.2">V2.2 — Hybrid Retrieval (Recommended)</option>
            <option value="v3">V3 — Structural RAG (PDF Layout + Expansion)</option>
            <option value="v2.1">V2.1 — BM25 Lexical</option>
            <option value="v1">V1 — Dense Baseline (Frozen)</option>
          </select>

          {/* Backend Strategy Info Badge for V3 */}
          {version === "v3" && (
            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-purple-950/80 text-purple-200 border border-purple-700/80">
              Backend strategy: Auto (Multi-Strategy)
            </span>
          )}

          <button
            type="button"
            onClick={() => setShowInfo(!showInfo)}
            className="p-1 text-slate-400 hover:text-white transition-colors ml-1"
            title="System Version & Pipeline Information"
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
      {chatDocuments.length > 0 && (
        <div className="mt-3 bg-purple-600/10 border border-purple-500/20 px-3 py-1.5 rounded-xl flex items-center justify-between text-xs text-purple-300">
          <span className="flex items-center gap-1.5 font-medium truncate">
            <Sparkles className="w-3.5 h-3.5 text-purple-400 shrink-0" />
            Active Chat Document: <strong className="text-white">{chatDocuments[0].filename}</strong>
          </span>
          <span className="text-[10px] text-purple-400 uppercase font-bold bg-purple-500/20 px-1.5 py-0.5 rounded">
            Chat Scoped
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
              Retrieving context via {version.toUpperCase()} search & synthesizing response...
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
            placeholder={`Ask NEXUS a question using ${version.toUpperCase()} pipeline...`}
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
