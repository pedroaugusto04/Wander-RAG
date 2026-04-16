"""Prompt templates for the Wander Jr RAG pipeline."""

SYSTEM_PROMPT = """\
Você é o Wander Jr, assistente virtual do CEFET-MG campus Timóteo.
Sua missão é ajudar os estudantes tirando dúvidas sobre a instituição. Use um tom amigável, parceiro e direto, como se estivesse conversando no Telegram.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nas informações do CONTEXTO abaixo. Nunca invente informações.
2. Se não visualizar a resposta no contexto, diga claramente que não sabe e oriente o aluno a buscar a secretaria.
3. Se couber de forma fluida, mencione a fonte (ex: "Segundo o Manual do Aluno...").
4. Se a pergunta for íntima/específica (notas, faltas, histórico), avise que não tem acesso a isso e peça para o aluno olhar o sistema acadêmico oficial (SIGAA).
5. Mantenha os textos curtos e escaneáveis. Evite formalidades exageradas e respostas longas.
"""

CONTEXT_TEMPLATE = """\
## CONTEXTO (documentos institucionais relevantes):

{retrieved_chunks}

## HISTÓRICO DA CONVERSA:
{conversation_history}

## PERGUNTA DO USUÁRIO:
{user_question}
"""

NO_CONTEXT_RESPONSE = (
    "Desculpe! Não achei nenhuma informação oficial sobre esse assunto na minha base de dados.\n\n"
    "Recomendo entrar em contato com a secretaria do campus Timóteo ou conferir o site oficial do CEFET-MG."
)

LOW_CONFIDENCE_DISCLAIMER = (
    "*Com base nas informações disponíveis, esta é minha melhor resposta, "
    "mas recomendo confirmar com o setor responsável:*\n\n"
)

FALLBACK_ERROR_RESPONSE = (
    "Desculpe, estou com dificuldades técnicas no momento. "
    "Por favor, tente novamente em alguns instantes.\n\n"
    "Se o problema persistir, entre em contato com a secretaria."
)


def build_rag_prompt(
    user_question: str,
    retrieved_chunks: list[dict[str, str]],
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the full prompt with system instruction, context, and question.

    Returns a list of messages in the format expected by LLMProvider.generate().
    """
    if retrieved_chunks:
        chunks_text = "\n\n---\n\n".join(
            f"**Fonte: {chunk.get('source', 'Documento institucional')}**\n{chunk['content']}"
            for chunk in retrieved_chunks
        )
    else:
        chunks_text = "(Nenhum documento relevante encontrado)"

    history_text = "(Início da conversa)"
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:  # Last 6 turns max
            role = "Usuário" if turn["role"] == "user" else "Wander Jr"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    user_content = CONTEXT_TEMPLATE.format(
        retrieved_chunks=chunks_text,
        conversation_history=history_text,
        user_question=user_question,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
