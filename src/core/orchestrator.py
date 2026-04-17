"""AI Orchestrator — coordinates RAG pipeline, LLM, and post-processing."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from src.ai.rag.prompts import (
    FALLBACK_ERROR_RESPONSE,
    GENERAL_GUIDANCE_DISCLAIMER,
    LOW_CONFIDENCE_DISCLAIMER,
    SENSITIVE_DATA_RESPONSE,
    build_general_guidance_prompt,
    build_grounded_answer_prompt,
)
from src.config.settings import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE

if TYPE_CHECKING:
    from src.ai.llm.base import LLMProvider, LLMResponse
    from src.ai.rag.pipeline import RAGPipeline
    from src.core.models import ConversationContext, IncomingMessage

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """Central coordinator: receives a message, runs RAG + LLM, returns response."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        rag_pipeline: RAGPipeline,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ) -> None:
        self.llm = llm_provider
        self.rag = rag_pipeline
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def process(
        self,
        message: IncomingMessage,
        context: ConversationContext | None = None,
    ) -> str:
        """Compatibility wrapper returning only response text."""
        response_text, _metadata = await self.process_with_metadata(message, context)
        return response_text

    async def process_with_metadata(
        self,
        message: IncomingMessage,
        context: ConversationContext | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Process a user message through the full AI pipeline.

        Flow:
        1. Run RAG retrieval
        2. Evaluate confidence
        3. Call LLM if context found
        4. Post-process response
        5. Log metrics
        """
        start_time = time.monotonic()

        try:
            conversation_history = context.get_recent_history() if context else None
            # ConversationManager stores the current user turn before calling the orchestrator.
            # Remove it from history to avoid duplicating the same question in the prompt.
            if (
                conversation_history
                and conversation_history[-1].get("role") == "user"
                and conversation_history[-1].get("content", "").strip() == message.text.strip()
            ):
                conversation_history = conversation_history[:-1]

            rag_result = await self.rag.process(
                query=message.text,
                conversation_history=conversation_history,
            )

            confidence: str = rag_result["confidence"]
            max_score: float = rag_result["max_score"]
            retrieval_query: str = rag_result["retrieval_query"]
            retrieved_chunks_count = len(rag_result["chunks"])

            if self._is_sensitive_academic_query(message.text):
                response_metadata = {
                    "response_mode": "sensitive_refusal",
                    "retrieval_confidence": confidence,
                    "retrieval_query": retrieval_query,
                    "retrieved_chunks_count": retrieved_chunks_count,
                }
                self._log_interaction(
                    message,
                    None,
                    confidence,
                    max_score,
                    start_time,
                    response_text=SENSITIVE_DATA_RESPONSE,
                    retrieval_query=retrieval_query,
                    response_mode="sensitive_refusal",
                    retrieved_chunks_count=retrieved_chunks_count,
                )
                return SENSITIVE_DATA_RESPONSE, response_metadata

            response_mode = "grounded"
            messages = build_grounded_answer_prompt(
                user_question=message.text,
                retrieved_chunks=rag_result["retrieved_chunks"],
                conversation_history=conversation_history,
                allow_general_guidance=confidence == "low",
                assistant_name=self.rag.assistant_name,
                institution_name=self.rag.institution_name,
                max_history_turns=self.rag.prompt_history_turns,
            )

            if confidence == "low":
                response_mode = "grounded_with_general_guidance"
            elif confidence == "none":
                response_mode = "general_guidance_only"
                messages = build_general_guidance_prompt(
                    user_question=message.text,
                    conversation_history=conversation_history,
                    assistant_name=self.rag.assistant_name,
                    institution_name=self.rag.institution_name,
                    max_history_turns=self.rag.prompt_history_turns,
                )

            llm_response: LLMResponse = await self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response_text = llm_response.content.strip()

            if confidence == "low":
                response_text = LOW_CONFIDENCE_DISCLAIMER + response_text
            elif confidence == "none":
                response_text = GENERAL_GUIDANCE_DISCLAIMER + response_text

            self._log_interaction(
                message,
                llm_response,
                confidence,
                max_score,
                start_time,
                response_text=response_text,
                retrieval_query=retrieval_query,
                response_mode=response_mode,
                retrieved_chunks_count=retrieved_chunks_count,
            )

            response_metadata: dict[str, Any] = {
                "model_used": llm_response.model,
                "token_usage": llm_response.usage,
                "response_mode": response_mode,
                "retrieval_confidence": confidence,
                "retrieval_query": retrieval_query,
                "retrieved_chunks_count": retrieved_chunks_count,
            }

            return response_text, response_metadata

        except Exception as exc:
            logger.error("Error in AI orchestrator: %s", exc)
            return FALLBACK_ERROR_RESPONSE, None

    def _log_interaction(
        self,
        message: IncomingMessage,
        llm_response: LLMResponse | None,
        confidence: str,
        max_score: float,
        start_time: float,
        response_text: str | None = None,
        retrieval_query: str | None = None,
        response_mode: str | None = None,
        retrieved_chunks_count: int | None = None,
    ) -> None:
        """Log interaction metrics for monitoring."""
        latency_ms = (time.monotonic() - start_time) * 1000

        metrics: dict[str, Any] = {
            "channel": message.channel.value,
            "user_question": message.text[:100],
            "confidence": confidence,
            "max_relevance_score": round(max_score, 3),
            "latency_ms": round(latency_ms, 1),
        }

        if retrieval_query:
            metrics["retrieval_query"] = retrieval_query[:160]
        if response_mode:
            metrics["response_mode"] = response_mode
        if retrieved_chunks_count is not None:
            metrics["retrieved_chunks_count"] = retrieved_chunks_count

        if llm_response:
            metrics.update(
                {
                    "llm_model": llm_response.model,
                    "prompt_tokens": llm_response.usage.get("prompt_tokens", 0),
                    "completion_tokens": llm_response.usage.get("completion_tokens", 0),
                    "finish_reason": llm_response.finish_reason,
                }
            )

        if response_text:
            compact_response = " ".join(response_text.split())
            metrics["assistant_response"] = compact_response[:280]

        logger.info("Interaction: %s", metrics)

    @staticmethod
    def _is_sensitive_academic_query(text: str) -> bool:
        """Detect requests about individual academic/private data."""
        normalized = " ".join(text.lower().split())
        possessive_markers = ("meu ", "minha ", "meus ", "minhas ")
        sensitive_terms = (
            "nota",
            "notas",
            "falta",
            "faltas",
            "histórico",
            "historico",
            "frequência",
            "frequencia",
            "ira",
            "cr",
            "dados pessoais",
        )

        if "dados pessoais" in normalized:
            return True

        return any(marker in normalized for marker in possessive_markers) and any(
            term in normalized for term in sensitive_terms
        )
