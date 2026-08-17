"use client";

import { useState } from "react";
import { DocumentMetadata, uploadDocument, deleteDocument } from "@/lib/api";
import {
  UploadCloud,
  FileText,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Filter,
  Check,
  FolderOpen,
} from "lucide-react";

interface DocumentSidebarProps {
  documents: DocumentMetadata[];
  selectedDocId: string | null;
  onSelectDoc: (docId: string | null) => void;
  onRefresh: () => void;
}

export default function DocumentSidebar({
  documents,
  selectedDocId,
  onSelectDoc,
  onRefresh,
}: DocumentSidebarProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    // Validate file type
    const allowedExts = ["pdf", "docx", "doc", "txt", "md"];
    const ext = file.name.split(".").pop()?.toLowerCase() || "";
    if (!allowedExts.includes(ext)) {
      setError(`Unsupported file format '.${ext}'. Allowed: PDF, DOCX, TXT, MD.`);
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);
    try {
      const doc = await uploadDocument(file);
      setSuccess(`Ingested '${doc.filename}' (${doc.chunk_count} vector chunks)`);
      onRefresh();
    } catch (err: any) {
      setError(err.message || "Document upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDelete = async (docId: string) => {
    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      if (selectedDocId === docId) {
        onSelectDoc(null);
      }
      setConfirmDeleteId(null);
      onRefresh();
    } catch (err: any) {
      setError(`Failed to delete document: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <aside className="glass-panel rounded-2xl p-5 flex flex-col h-full space-y-5 border border-slate-800/80">
      {/* Sidebar Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FolderOpen className="w-5 h-5 text-blue-400" />
          <h2 className="text-sm font-bold text-white tracking-wide">Document Repository</h2>
        </div>
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          title="Refresh document list"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-5 text-center transition-all ${
          uploading
            ? "border-blue-500/50 bg-blue-500/5 cursor-not-allowed"
            : "border-slate-800 hover:border-blue-500/50 hover:bg-slate-800/30"
        }`}
      >
        <input
          type="file"
          id="file-upload-input"
          className="hidden"
          accept=".pdf,.docx,.doc,.txt,.md"
          disabled={uploading}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
        <label htmlFor="file-upload-input" className="cursor-pointer block">
          {uploading ? (
            <div className="flex flex-col items-center justify-center py-2 space-y-2">
              <Loader2 className="w-7 h-7 text-blue-400 animate-spin" />
              <p className="text-xs font-semibold text-blue-300">Extracting & Vectorizing...</p>
              <p className="text-[10px] text-slate-400">Embedding vectors into Qdrant</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <UploadCloud className="w-5 h-5" />
              </div>
              <p className="text-xs font-medium text-slate-200">
                Drag document or <span className="text-blue-400 underline">browse</span>
              </p>
              <p className="text-[10px] text-slate-400">PDF, DOCX, TXT, MD up to 25MB</p>
            </div>
          )}
        </label>
      </div>

      {/* Feedback Messages */}
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

      {/* Document Selection / List */}
      <div className="flex-1 flex flex-col min-h-0 space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
            Uploaded Files ({documents.length})
          </span>
          {selectedDocId && (
            <button
              onClick={() => onSelectDoc(null)}
              className="text-[10px] text-blue-400 hover:underline flex items-center gap-1"
            >
              <Filter className="w-3 h-3" /> Clear Filter
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {documents.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-xs italic">
              No documents uploaded yet. Drop a PDF or text file above.
            </div>
          ) : (
            documents.map((doc) => {
              const isSelected = selectedDocId === doc.document_id;
              const isConfirming = confirmDeleteId === doc.document_id;

              return (
                <div
                  key={doc.document_id}
                  onClick={() => onSelectDoc(isSelected ? null : doc.document_id)}
                  className={`p-3 rounded-xl cursor-pointer border transition-all ${
                    isSelected
                      ? "bg-blue-600/15 border-blue-500/50 shadow-md shadow-blue-500/10"
                      : "glass-panel-interactive border-slate-800"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center space-x-2.5 min-w-0">
                      <div
                        className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border ${
                          isSelected
                            ? "bg-blue-600 text-white border-blue-500"
                            : "bg-slate-800 text-blue-400 border-slate-700"
                        }`}
                      >
                        <FileText className="w-4 h-4" />
                      </div>
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-slate-200 truncate">
                          {doc.filename}
                        </p>
                        <div className="flex items-center gap-1.5 text-[10px] text-slate-400 mt-0.5">
                          <span className="uppercase text-blue-400 font-bold">{doc.file_type}</span>
                          <span>•</span>
                          <span>{doc.chunk_count} chunks</span>
                          <span>•</span>
                          <span>{(doc.char_count / 1000).toFixed(1)}k chars</span>
                        </div>
                      </div>
                    </div>

                    {/* Delete Controls */}
                    <div className="flex items-center space-x-1 shrink-0" onClick={(e) => e.stopPropagation()}>
                      {isConfirming ? (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => handleDelete(doc.document_id)}
                            disabled={deletingId === doc.document_id}
                            className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white hover:bg-rose-500"
                          >
                            {deletingId === doc.document_id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              "Delete"
                            )}
                          </button>
                          <button
                            onClick={() => setConfirmDeleteId(null)}
                            className="px-1.5 py-0.5 rounded text-[10px] text-slate-400 hover:text-white"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setConfirmDeleteId(doc.document_id)}
                          className="p-1 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-md transition-colors"
                          title="Delete document"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>

                  {isSelected && (
                    <div className="mt-2 pt-2 border-t border-blue-500/20 flex items-center justify-between text-[10px] text-blue-300">
                      <span className="flex items-center gap-1">
                        <Check className="w-3 h-3 text-blue-400" /> Active Scope Filter
                      </span>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
