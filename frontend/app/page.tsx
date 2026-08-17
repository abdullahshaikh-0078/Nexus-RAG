"use client";

import { useEffect, useState, useCallback } from "react";
import Navbar from "@/components/Navbar";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatInterface, { Message } from "@/components/ChatInterface";
import { listDocuments, DocumentMetadata } from "@/lib/api";

const INITIAL_WELCOME_MESSAGE: Message = {
  id: "welcome",
  sender: "assistant",
  text: "Hello! I am NEXUS RAG. Upload your documents in the sidebar to extract text, chunk content, generate vector embeddings, and store vectors in Qdrant.\n\nAsk any question, and I will retrieve relevant context chunks and synthesize an accurate answer with source citations.",
};

export default function Home() {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([INITIAL_WELCOME_MESSAGE]);

  const loadDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error("Failed to load documents list:", err);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleNewChat = () => {
    setMessages([
      {
        ...INITIAL_WELCOME_MESSAGE,
        id: `welcome-${Date.now()}`,
      },
    ]);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar onNewChat={handleNewChat} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-5rem)]">
        {/* Left Column: Document Sidebar (4 cols on lg) */}
        <div className="lg:col-span-4 h-full">
          <DocumentSidebar
            documents={documents}
            selectedDocId={selectedDocId}
            onSelectDoc={(id) => setSelectedDocId(id)}
            onRefresh={loadDocuments}
          />
        </div>

        {/* Right Column: Main RAG Conversation Area (8 cols on lg) */}
        <div className="lg:col-span-8 h-full">
          <ChatInterface
            documents={documents}
            selectedDocId={selectedDocId}
            messages={messages}
            setMessages={setMessages}
          />
        </div>
      </main>
    </div>
  );
}
