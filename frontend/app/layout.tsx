import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEXUS RAG - Enterprise Retrieval Augmented Generation Platform",
  description: "Production-grade document search, vector storage with Qdrant, and context-aware LLM synthesis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#0b0f19] text-slate-100 selection:bg-blue-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
