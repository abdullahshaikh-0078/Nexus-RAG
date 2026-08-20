"use client";

import { DocumentMetadata } from "@/lib/api";
import { FileText } from "lucide-react";

interface DocumentManagerProps {
  documents?: DocumentMetadata[];
  onRefresh?: () => void;
}

export default function DocumentManager({ documents = [], onRefresh }: DocumentManagerProps) {
  return (
    <div className="glass-panel rounded-2xl p-6 flex flex-col h-full space-y-6">
      <div>
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-400" /> Chat Document Manager
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Chat-scoped document model active.
        </p>
      </div>
    </div>
  );
}
