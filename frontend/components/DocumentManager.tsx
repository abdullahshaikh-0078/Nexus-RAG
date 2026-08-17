"use client";

import { useState } from "react";
import { DocumentMetadata, uploadDocument, deleteDocument } from "@/lib/api";
import { UploadCloud, FileText, Trash2, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

interface DocumentManagerProps {
  documents: DocumentMetadata[];
  onRefresh: () => void;
}

export default function DocumentManager({ documents, onRefresh }: DocumentManagerProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    setSuccess(null);
    try {
      const doc = await uploadDocument(file);
      setSuccess(`Successfully ingested '${doc.filename}' (${doc.chunk_count} vector chunks)`);
      onRefresh();
    } catch (err: any) {
      setError(err.message || "Failed to upload document");
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

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete '${filename}'?`)) return;
    setDeletingId(docId);
    try {
      await deleteDocument(docId);
      onRefresh();
    } catch (err: any) {
      alert(`Delete error: ${err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6 flex flex-col h-full space-y-6">
      <div>
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-400" /> Document Repository
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload PDF, DOCX, TXT, or MD files to extract, chunk, embed, and store in Qdrant.
        </p>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
          uploading
            ? "border-blue-500/50 bg-blue-500/5 cursor-not-allowed"
            : "border-slate-700 hover:border-blue-500/50 hover:bg-slate-800/40"
        }`}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".pdf,.docx,.doc,.txt,.md"
          disabled={uploading}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />
        <label htmlFor="file-upload" className="cursor-pointer block">
          {uploading ? (
            <div className="flex flex-col items-center justify-center py-2 space-y-2">
              <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
              <p className="text-sm font-medium text-blue-300">Extracting & Vectorizing Document...</p>
              <p className="text-xs text-slate-500">Generating SentenceTransformers embeddings</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center space-y-2">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <UploadCloud className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Drop document here or <span className="text-blue-400 underline">browse</span>
              </p>
              <p className="text-[11px] text-slate-400">PDF, DOCX, TXT, MD up to 25MB</p>
            </div>
          )}
        </label>
      </div>

      {/* Notification Feedback */}
      {error && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-300">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{success}</span>
        </div>
      )}

      {/* Uploaded Documents List */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Ingested Documents ({documents.length})
          </span>
        </div>

        <div className="flex-1 overflow-y-auto mt-3 space-y-2 pr-1">
          {documents.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-xs italic">
              No documents ingested yet. Upload a file above to begin.
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.document_id}
                className="glass-panel-interactive p-3 rounded-xl flex items-center justify-between group"
              >
                <div className="flex items-center space-x-3 min-w-0">
                  <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center shrink-0 text-blue-400 border border-slate-700">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-200 truncate">{doc.filename}</p>
                    <div className="flex items-center gap-2 text-[11px] text-slate-400 mt-0.5">
                      <span className="uppercase text-blue-400 font-semibold">{doc.file_type}</span>
                      <span>•</span>
                      <span>{doc.chunk_count} chunks</span>
                      <span>•</span>
                      <span>{(doc.char_count / 1000).toFixed(1)}k chars</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleDelete(doc.document_id, doc.filename)}
                  disabled={deletingId === doc.document_id}
                  className="p-1.5 text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
                  title="Delete Document"
                >
                  {deletingId === doc.document_id ? (
                    <Loader2 className="w-4 h-4 animate-spin text-rose-400" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
