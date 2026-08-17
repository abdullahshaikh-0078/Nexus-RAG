"use client";

import { useState, useRef, useEffect } from "react";
import { queryRag, ChatQueryResponse, SourceCitation, DocumentMetadata } from "@/lib/api";
import SourceCitations from "./SourceCitations";
import SourceInspector from "./SourceInspector";
import { Send, Bot, User, Sparkles, Clock, Cpu, Loader2, AlertTriangle, CornerDownLeft } from "lucide-react";

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<SourceCitation | null>(null);

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
      const res = await queryRag(query, 4, docIdsFilter);
      
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

  const activeDoc = documents.find((d) => d.document_id === selectedDocId);

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col h-full space-y-4 border border-slate-800/80">
      {/* Scope Filter Status */}
      {selectedDocId && activeDoc && (
        <div className="bg-blue-600/10 border border-blue-500/20 px-3 py-1.5 rounded-xl flex items-center justify-between text-xs text-blue-300">
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
      <div className="flex-1 overflow-y-auto space-y-5 pr-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3">
            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
              <Bot className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-200">Start a RAG Conversation</h3>
            <p className="text-xs text-slate-400 max-w-md">
              Ask questions about your uploaded documents. NEXUS RAG retrieves semantic vector context from Qdrant and synthesizes grounded answers.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.sender === "assistant" && (
                <div className="w-8 h-8 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
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

                  {/* Metadata and Sources for assistant response */}
                  {msg.responseMeta && (
                    <>
                      <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
                        <span className="flex items-center gap-1">
                          <Cpu className="w-3 h-3 text-blue-400" /> {msg.responseMeta.llm_provider} ({msg.responseMeta.model_name})
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3 text-indigo-400" /> {msg.responseMeta.processing_time_seconds}s
                        </span>
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
            <div className="w-8 h-8 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="glass-panel p-3.5 rounded-2xl text-xs text-indigo-300 flex items-center gap-2 border border-slate-800">
              <Sparkles className="w-4 h-4 animate-pulse text-blue-400" />
              Searching Qdrant vectors & synthesizing response...
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

      {/* Fixed Bottom Input Area */}
      <div className="pt-2 border-t border-slate-800/80">
        <div className="relative flex items-center">
          <textarea
            ref={textareaRef}
            rows={2}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              documents.length > 0
                ? "Ask a question about your documents... (Enter to send, Shift+Enter for newline)"
                : "Upload a document in the sidebar to begin RAG search..."
            }
            disabled={loading}
            className="w-full pl-4 pr-12 py-3 text-xs bg-slate-900 border border-slate-800 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/80 transition-colors resize-none disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!inputQuery.trim() || loading}
            className="absolute right-3 p-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white transition-all shadow-md shadow-blue-600/20"
            title="Send query"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <div className="flex items-center justify-between mt-1.5 px-1 text-[10px] text-slate-500">
          <span className="flex items-center gap-1">
            <CornerDownLeft className="w-2.5 h-2.5" /> Press <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-400">Shift+Enter</kbd> for newline
          </span>
          <span>NEXUS RAG V1.5</span>
        </div>
      </div>

      {/* Citation Inspector Modal */}
      <SourceInspector
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
