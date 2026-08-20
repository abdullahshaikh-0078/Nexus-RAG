"use client";

import { useEffect, useState, useCallback } from "react";
import Navbar from "@/components/Navbar";
import DocumentSidebar from "@/components/DocumentSidebar";
import ChatInterface, { Message } from "@/components/ChatInterface";
import {
  createChat,
  listChats,
  getChatDetail,
  listChatDocuments,
  ChatSession,
  ChatDocument,
} from "@/lib/api";

const INITIAL_WELCOME_MESSAGE: Message = {
  id: "welcome",
  sender: "assistant",
  text: "Hello! I am NEXUS RAG. Upload a document to this chat session to extract text, chunk content, generate vector embeddings, and perform intelligent RAG search.\n\nSelect pipeline version V1 (Dense), V2.1 (BM25), V2.2 (Hybrid RRF), or click 'Convert PDF to V3' to execute explicit V3 layout-aware structural RAG.",
};

export default function Home() {
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatDocuments, setChatDocuments] = useState<ChatDocument[]>([]);
  const [activeVersion, setActiveVersion] = useState<"v1" | "v2.1" | "v2.2" | "v3">("v1");
  const [messages, setMessages] = useState<Message[]>([INITIAL_WELCOME_MESSAGE]);

  const loadChatState = useCallback(async (chatId: string) => {
    try {
      const detail = await getChatDetail(chatId);
      setChatDocuments(detail.documents || []);
      if (detail.chat.active_version) {
        setActiveVersion(detail.chat.active_version as any);
      }
    } catch (err) {
      console.error("Failed to load chat detail:", err);
    }
  }, []);

  const handleCreateNewChat = useCallback(async () => {
    try {
      const newChat = await createChat("New Chat");
      setActiveChatId(newChat.chat_id);
      setChatDocuments([]);
      setActiveVersion("v1");
      setMessages([
        {
          ...INITIAL_WELCOME_MESSAGE,
          id: `welcome-${Date.now()}`,
        },
      ]);
    } catch (err) {
      console.error("Failed to create new chat:", err);
    }
  }, []);

  useEffect(() => {
    async function initChats() {
      try {
        const chats = await listChats();
        if (chats.length > 0) {
          setActiveChatId(chats[0].chat_id);
          await loadChatState(chats[0].chat_id);
        } else {
          await handleCreateNewChat();
        }
      } catch (err) {
        await handleCreateNewChat();
      }
    }
    initChats();
  }, []);

  const handleSelectChat = async (chatId: string) => {
    setActiveChatId(chatId);
    setMessages([
      {
        ...INITIAL_WELCOME_MESSAGE,
        id: `welcome-${Date.now()}`,
      },
    ]);
    await loadChatState(chatId);
  };

  const handleRefresh = useCallback(async () => {
    if (activeChatId) {
      await loadChatState(activeChatId);
    }
  }, [activeChatId, loadChatState]);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100">
      <Navbar onNewChat={handleCreateNewChat} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-5rem)]">
        {/* Left Column: Document & Chat Sidebar */}
        <div className="lg:col-span-4 h-full">
          <DocumentSidebar
            activeChatId={activeChatId}
            onSelectChat={handleSelectChat}
            onNewChat={handleCreateNewChat}
            chatDocuments={chatDocuments}
            activeVersion={activeVersion}
            onVersionChange={(v) => setActiveVersion(v)}
            onRefresh={handleRefresh}
          />
        </div>

        {/* Right Column: Main RAG Conversation Area */}
        <div className="lg:col-span-8 h-full">
          <ChatInterface
            activeChatId={activeChatId}
            chatDocuments={chatDocuments}
            version={activeVersion}
            setVersion={setActiveVersion}
            messages={messages}
            setMessages={setMessages}
          />
        </div>
      </main>
    </div>
  );
}
