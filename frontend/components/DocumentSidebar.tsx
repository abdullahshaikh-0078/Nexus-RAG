"use client";

import { useState, useEffect } from "react";
import {
  ChatSession,
  ChatDocument,
  DocumentRepresentation,
  listChats,
  createChat,
  deleteChat,
  uploadChatDocument,
  fetchChatDocumentRepresentations,
  convertChatDocumentToV3,
} from "@/lib/api";
import {
  UploadCloud,
  FileText,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Plus,
  MessageSquare,
  Layers,
  Sparkles,
  Zap,
} from "lucide-react";

interface DocumentSidebarProps {
  activeChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  chatDocuments: ChatDocument[];
  activeVersion: string;
  onVersionChange: (version: "v1" | "v2.1" | "v2.2" | "v3") => void;
  onRefresh: () => void;
}

export default function DocumentSidebar({
  activeChatId,
  onSelectChat,
  onNewChat,
  chatDocuments,
  activeVersion,
  onVersionChange,
  onRefresh,
}: DocumentSidebarProps) {
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [uploading, setUploading] = useState(false);
  const [convertingDocId, setConvertingDocId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deletingChatId, setDeletingChatId] = useState<string | null>(null);
  const [repsMap, setRepsMap] = useState<Record<string, DocumentRepresentation[]>>({});

  useEffect(() => {
    async function loadChats() {
      try {
        const chatList = await listChats();
        setChats(chatList);
      } catch (err) {
        // ignore polling error
      }
    }
    loadChats();
  }, [activeChatId]);

  useEffect(() => {
    async function loadRepresentations() {
      if (!activeChatId || chatDocuments.length === 0) {
        setRepsMap({});
        return;
      }
      const newMap: Record<string, DocumentRepresentation[]> = {};
      for (const doc of chatDocuments) {
        try {
          const res = await fetchChatDocumentRepresentations(activeChatId, doc.document_id);
          newMap[doc.document_id] = res.representations;
        } catch (err) {
          // ignore error
        }
      }
      setRepsMap(newMap);
    }
    loadRepresentations();
  }, [activeChatId, chatDocuments]);

  const handleFileUpload = async (file: File) => {
    if (!activeChatId) {
      setError("Please create or select a chat session first.");
      return;
    }

    const allowedExts = ["pdf", "docx", "doc", "txt", "md"];
    const ext = file.name.split(".").pop()?.toLowerCase() || "";
    if (!allowedExts.includes(ext)) {
      setError(`Unsupported format '.${ext}'. Allowed: PDF, DOCX, TXT, MD.`);
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);
    try {
      const doc = await uploadChatDocument(activeChatId, file);
      setSuccess(`Uploaded '${doc.filename}' (${doc.v1_chunk_count} V1 chunks)`);
      onRefresh();
    } catch (err: any) {
      setError(err.message || "Document upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleTriggerV3Convert = async (docId: string) => {
    if (!activeChatId) return;

    setConvertingDocId(docId);
    setError(null);
    setSuccess(null);

    try {
      const res = await convertChatDocumentToV3(activeChatId, docId);
      if (res.success) {
        setSuccess(`V3 Structural Conversion complete! ${res.representation.chunk_count} layout-aware chunks generated.`);
        onVersionChange("v3");
        onRefresh();
      } else {
        setError(res.representation.error_message || "V3 Conversion failed.");
      }
    } catch (err: any) {
      setError(err.message || "Failed to convert PDF to V3.");
    } finally {
      setConvertingDocId(null);
    }
  };

  const handleDeleteChatSession = async (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingChatId(chatId);
    try {
      await deleteChat(chatId);
      setChats((prev) => prev.filter((c) => c.chat_id !== chatId));
      if (activeChatId === chatId) {
        onNewChat();
      }
    } catch (err: any) {
      setError(`Failed to delete chat: ${err.message}`);
    } finally {
      setDeletingChatId(null);
    }
  };

  const activeDoc = chatDocuments[0];
  const activeV3Rep = activeDoc ? repsMap[activeDoc.document_id]?.find((r) => r.version === "v3") : undefined;

  return (
    <aside className="glass-panel rounded-2xl p-5 flex flex-col h-full space-y-4 border border-slate-800/80">
      {/* New Chat Button */}
      <button
        onClick={onNewChat}
        className="w-full py-2.5 px-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white rounded-xl font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-purple-600/20 transition-all"
      >
        <Plus className="w-4 h-4" />
        New Chat
      </button>

      {/* Chat History Switcher */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-1">
          Chat Sessions ({chats.length})
        </span>
        <div className="max-h-32 overflow-y-auto space-y-1 pr-1">
          {chats.map((c) => {
            const isActive = c.chat_id === activeChatId;
            return (
              <div
                key={c.chat_id}
                onClick={() => onSelectChat(c.chat_id)}
                className={`px-3 py-2 rounded-xl text-xs flex items-center justify-between cursor-pointer transition-all ${
                  isActive
                    ? "bg-purple-600/20 border border-purple-500/50 text-white font-semibold"
                    : "glass-panel-interactive text-slate-300 border-slate-800/60"
                }`}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <MessageSquare className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                  <span className="truncate">{c.title || "New Chat"}</span>
                </div>
                <button
                  onClick={(e) => handleDeleteChatSession(c.chat_id, e)}
                  disabled={deletingChatId === c.chat_id}
                  className="p-1 text-slate-500 hover:text-rose-400 transition-colors"
                  title="Delete chat session"
                >
                  {deletingChatId === c.chat_id ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Trash2 className="w-3 h-3" />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Upload Zone for Current Chat */}
      <div className="pt-2 border-t border-slate-800/80">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-1 block mb-2">
          Chat Document
        </span>

        {chatDocuments.length === 0 ? (
          <div className="border-2 border-dashed border-slate-800 hover:border-purple-500/50 rounded-xl p-4 text-center transition-all bg-slate-900/40">
            <input
              type="file"
              id="chat-file-upload"
              className="hidden"
              accept=".pdf,.docx,.doc,.txt,.md"
              disabled={uploading}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileUpload(e.target.files[0]);
                }
              }}
            />
            <label htmlFor="chat-file-upload" className="cursor-pointer block">
              {uploading ? (
                <div className="py-2 space-y-1.5">
                  <Loader2 className="w-6 h-6 text-purple-400 animate-spin mx-auto" />
                  <p className="text-xs font-semibold text-purple-300">Ingesting PDF (V1 Baseline)...</p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  <UploadCloud className="w-6 h-6 text-purple-400 mx-auto" />
                  <p className="text-xs font-medium text-slate-200">
                    Upload PDF for this chat
                  </p>
                  <p className="text-[10px] text-slate-400">PDF, DOCX up to 25MB</p>
                </div>
              )}
            </label>
          </div>
        ) : (
          /* Uploaded Chat Document Card */
          <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-purple-400 shrink-0" />
              <span className="text-xs font-semibold text-slate-200 truncate">
                {activeDoc.filename}
              </span>
            </div>
            <div className="text-[10px] text-slate-400 space-y-1">
              <div>V1 Original: {activeDoc.v1_chunk_count} chunks</div>
              {activeV3Rep && activeV3Rep.status === "READY" && (
                <div className="text-purple-300 font-medium flex items-center gap-1">
                  <Layers className="w-3 h-3 text-purple-400" />
                  V3 {activeV3Rep.chunking_strategy || "table_aware"}: {activeV3Rep.chunk_count} chunks ✓
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* EXPLICIT V3 CONVERSION ACTION BANNER */}
      {chatDocuments.length > 0 && activeVersion === "v3" && (!activeV3Rep || activeV3Rep.status !== "READY") && (
        <div className="p-3.5 rounded-xl bg-purple-950/40 border border-purple-500/40 space-y-2.5 animate-fadeIn">
          <div className="flex items-center gap-1.5 text-xs font-bold text-purple-200">
            <Sparkles className="w-4 h-4 text-purple-400 animate-pulse" />
            Explicit V3 Structural Conversion Required
          </div>
          <p className="text-[11px] text-purple-300/90 leading-relaxed">
            V3 will structurally parse this PDF layout with PyMuPDF, preserve financial tables, and build a V3 structural retrieval representation.
          </p>
          <button
            onClick={() => handleTriggerV3Convert(activeDoc.document_id)}
            disabled={convertingDocId === activeDoc.document_id}
            className="w-full py-2 px-3 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-2 shadow-md transition-all"
          >
            {convertingDocId === activeDoc.document_id ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Converting PDF to V3 (PyMuPDF Layout Parsing)...
              </>
            ) : (
              <>
                <Zap className="w-3.5 h-3.5" />
                Convert PDF to V3
              </>
            )}
          </button>
        </div>
      )}

      {/* Messages / Alerts */}
      {error && (
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span className="truncate">{success}</span>
        </div>
      )}
    </aside>
  );
}

