import os
import logging
from typing import List, Tuple
from app.core.config import settings
from app.models.schemas import SourceCitation

logger = logging.getLogger(__name__)


class LLMService:
    """Manages generation of answers using Google Gemini, Groq, or Mock LLM."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.model_name = settings.LLM_MODEL_NAME

    def generate_answer(
        self, query: str, citations: List[SourceCitation]
    ) -> Tuple[str, str, str]:
        """
        Synthesizes an answer grounded in context citations.
        Returns: (answer_string, actual_provider_used, actual_model_used)
        """
        # Format context snippet string
        if citations:
            context_blocks = []
            for idx, cite in enumerate(citations, 1):
                context_blocks.append(
                    f"[{idx}] Source File: {cite.document_name} (Chunk #{cite.chunk_index})\n{cite.content}"
                )
            context_text = "\n\n".join(context_blocks)
        else:
            context_text = "No relevant context found in uploaded documents."

        system_prompt = (
            "You are NEXUS RAG, an intelligent and precise AI assistant.\n"
            "Your task is to answer the user's question accurately using ONLY the provided document context.\n"
            "If the context contains relevant information, answer clearly and reference the source documents.\n"
            "If the context does not contain enough information to answer the question, state politely that the "
            "uploaded documents do not contain sufficient context to answer."
        )

        user_prompt = f"DOCUMENT CONTEXT:\n{context_text}\n\nUSER QUESTION:\n{query}"

        # 1. Try Gemini if configured
        gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if self.provider == "gemini" or (gemini_key and self.provider not in ["groq"]):
            if gemini_key:
                try:
                    return self._call_gemini(system_prompt, user_prompt, gemini_key)
                except Exception as e:
                    logger.error(f"Gemini API invocation failed: {str(e)}")

        # 2. Try Groq if configured
        groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if self.provider == "groq" or (groq_key and self.provider == "groq"):
            if groq_key:
                try:
                    return self._call_groq(system_prompt, user_prompt, groq_key)
                except Exception as e:
                    logger.error(f"Groq API invocation failed: {str(e)}")

        # 3. Offline / Development Mock Fallback
        logger.info("Using Fallback Mock LLM Service (No valid API key provided or API error).")
        return self._call_mock(query, citations)

    def _call_gemini(self, system_prompt: str, user_prompt: str, api_key: str) -> Tuple[str, str, str]:
        """Calls Google Gemini API."""
        try:
            # Using standard google-genai SDK
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = client.models.generate_content(
                model=self.model_name or "gemini-1.5-flash",
                contents=prompt,
            )
            return response.text, "google-gemini", self.model_name or "gemini-1.5-flash"
        except ImportError:
            # Fallback to google-generativeai legacy package if installed
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name or "gemini-1.5-flash")
            response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return response.text, "google-gemini", self.model_name or "gemini-1.5-flash"

    def _call_groq(self, system_prompt: str, user_prompt: str, api_key: str) -> Tuple[str, str, str]:
        """Calls Groq API."""
        from groq import Groq
        client = Groq(api_key=api_key)
        model = self.model_name if "llama" in self.model_name or "mixtral" in self.model_name else "llama-3.1-8b-instant"
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content or ""
        return answer, "groq", model

    def _call_mock(self, query: str, citations: List[SourceCitation]) -> Tuple[str, str, str]:
        """Generates deterministic RAG mock response for offline development and testing."""
        if not citations:
            answer = f"No document context found for your query: '{query}'. Please upload a document to get started."
        else:
            top_source = citations[0]
            answer = (
                f"Based on **{top_source.document_name}**, here is what was found regarding '{query}':\n\n"
                f"\"{top_source.content[:300]}...\"\n\n"
                f"*(Retrieved {len(citations)} relevant chunk(s) with highest similarity score of {top_source.score})*"
            )
        return answer, "mock-provider", "nexus-v1-synthesizer"


llm_service = LLMService()
