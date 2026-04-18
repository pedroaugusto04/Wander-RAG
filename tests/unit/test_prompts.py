"""Tests for prompt building."""

from src.ai.rag.pipeline import RAGPipeline
from src.ai.rag.prompts import (
    CONTEXT_TEMPLATE,
    GENERAL_GUIDANCE_DISCLAIMER,
    SYSTEM_PROMPT,
    build_general_guidance_prompt,
    build_query_rewrite_prompt,
    build_rag_prompt,
)


class TestBuildRagPrompt:
    def test_basic_prompt_structure(self) -> None:
        messages = build_rag_prompt(
            user_question="Qual o horário da biblioteca?",
            retrieved_chunks=[
                {"content": "A biblioteca funciona de 8h às 18h.", "source": "Manual do Aluno"}
            ],
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "CEFET-MG" in messages[0]["content"]

    def test_includes_chunks(self) -> None:
        messages = build_rag_prompt(
            user_question="Prazo de matrícula?",
            retrieved_chunks=[
                {"content": "Matrícula vai até dia 15/03", "source": "Edital 2026"},
                {"content": "Documentos necessários: RG, CPF", "source": "Manual"},
            ],
        )
        user_msg = messages[1]["content"]
        assert "Matrícula vai até dia 15/03" in user_msg
        assert "Edital 2026" in user_msg

    def test_no_chunks(self) -> None:
        messages = build_rag_prompt(
            user_question="Algo sem contexto?",
            retrieved_chunks=[],
        )
        user_msg = messages[1]["content"]
        assert "Nenhum documento relevante" in user_msg

    def test_includes_history(self) -> None:
        messages = build_rag_prompt(
            user_question="E o valor?",
            retrieved_chunks=[{"content": "Mensalidade R$500", "source": "Doc"}],
            conversation_history=[
                {"role": "user", "content": "Quanto custa?"},
                {"role": "assistant", "content": "Depende do curso."},
            ],
        )
        user_msg = messages[1]["content"]
        assert "Quanto custa?" in user_msg
        assert "Depende do curso." in user_msg

    def test_system_prompt_content(self) -> None:
        assert "Wander Jr" in SYSTEM_PROMPT
        assert "CEFET-MG" in SYSTEM_PROMPT
        assert "Nunca invente" in SYSTEM_PROMPT
        assert "não encontrou essa informação nos documentos" in SYSTEM_PROMPT
        assert "o que está confirmado nos documentos" in SYSTEM_PROMPT
        assert "não encontrou confirmação nos documentos" in CONTEXT_TEMPLATE
        assert "Nunca use termos técnicos como \"contexto\"" in SYSTEM_PROMPT

    def test_general_guidance_prompt_mentions_orientation_only(self) -> None:
        messages = build_general_guidance_prompt(
            user_question="Como costuma funcionar trancamento?",
            conversation_history=[{"role": "user", "content": "Preciso de ajuda"}],
        )

        assert len(messages) == 2
        assert "orientação geral" in messages[0]["content"]
        assert "não encontrei confirmação específica nos documentos" in messages[1]["content"]
        assert "Não afirme fatos específicos" in messages[1]["content"]
        assert "Preciso de ajuda" in messages[1]["content"]

    def test_query_rewrite_prompt_requests_single_line_query(self) -> None:
        messages = build_query_rewrite_prompt(
            user_question="E o e-mail dele?",
            conversation_history=[{"role": "user", "content": "Quem é João Paulo de Castro Costa?"}],
        )

        assert len(messages) == 2
        assert "consulta independente" in messages[1]["content"]
        assert "João Paulo de Castro Costa" in messages[1]["content"]
        assert "uma única linha" in messages[0]["content"]

    def test_general_guidance_disclaimer_text_is_explicit(self) -> None:
        assert "orientar de forma geral" in GENERAL_GUIDANCE_DISCLAIMER
        assert "documentos oficiais" in GENERAL_GUIDANCE_DISCLAIMER
        assert "orientação inicial" in GENERAL_GUIDANCE_DISCLAIMER


class TestRagPipelineSourceLabels:
    def test_build_chunk_source_includes_section_breadcrumb(self) -> None:
        source = RAGPipeline._build_chunk_source(
            {
                "document_title": "docentes",
                "section_breadcrumb": "DECOM-TM > Professor: Adilson Mendes Ricardo",
            }
        )
        assert source == "docentes — DECOM-TM > Professor: Adilson Mendes Ricardo"

    def test_build_chunk_source_falls_back_to_title_only(self) -> None:
        source = RAGPipeline._build_chunk_source({"document_title": "manual"})
        assert source == "manual"
