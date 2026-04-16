"""AI Orchestrator — coordinates RAG pipeline, LLM, and post-processing."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from src.ai.rag.prompts import (
    FALLBACK_ERROR_RESPONSE,
    LOW_CONFIDENCE_DISCLAIMER,
    NO_CONTEXT_RESPONSE,
)

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
        temperature: float = 0.3,
        max_tokens: int = 1024,
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
            rag_result = await self.rag.process(
                query=message.text,
                conversation_history=context.get_recent_history() if context else None,
            )

            confidence: str = rag_result["confidence"]
            max_score: float = rag_result["max_score"]

            if confidence == "none":
                self._log_interaction(
                    message,
                    None,
                    confidence,
                    max_score,
                    start_time,
                    response_text=NO_CONTEXT_RESPONSE,
                )
                return NO_CONTEXT_RESPONSE, None

            llm_response: LLMResponse = await self.llm.generate(
                messages=rag_result["messages"],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response_text = llm_response.content.strip()

            if confidence == "low":
                response_text = LOW_CONFIDENCE_DISCLAIMER + response_text

            self._log_interaction(
                message,
                llm_response,
                confidence,
                max_score,
                start_time,
                response_text=response_text,
            )

            response_metadata: dict[str, Any] = {
                "model_used": llm_response.model,
                "token_usage": llm_response.usage,
            }

            return response_text, response_metadata

        except Exception:
            logger.exception("Error in AI orchestrator")
            return FALLBACK_ERROR_RESPONSE, None

    def _log_interaction(
        self,
        message: IncomingMessage,
        llm_response: LLMResponse | None,
        confidence: str,
        max_score: float,
        start_time: float,
        response_text: str | None = None,
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
