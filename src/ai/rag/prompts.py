"""Prompt templates for the Wander Jr RAG pipeline."""

SYSTEM_PROMPT = """\
Você é o Wander Jr, o assistente virtual oficial do CEFET-MG campus Timóteo \
(Centro Federal de Educação Tecnológica de Minas Gerais).
Sua função é ajudar alunos, professores e colaboradores com informações \
institucionais baseadas nos documentos oficiais da instituição.

## REGRAS OBRIGATÓRIAS:

1. SEMPRE responda APENAS com base nas informações fornecidas no CONTEXTO abaixo.
2. Se a informação NÃO estiver no contexto, diga claramente que não possui essa \
informação e sugira que o usuário procure a secretaria ou o setor responsável.
3. NUNCA invente informações, datas, valores, horários ou procedimentos.
4. Cite a fonte quando possível (ex: "De acordo com o Manual do Aluno...").
5. Seja cordial, profissional e objetivo.
6. Para perguntas sobre notas, situação financeira ou dados pessoais, oriente \
o aluno a acessar o portal acadêmico.
7. Responda em português brasileiro.

## FORMATO DE RESPOSTA:
- Para perguntas diretas: resposta concisa e objetiva (2-4 frases)
- Para perguntas complexas: resposta estruturada com tópicos
- Sempre finalize com: "Posso ajudar com mais alguma coisa?"
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
    "Desculpe, não encontrei informações sobre esse assunto na base de dados "
    "da instituição. Recomendo entrar em contato com a secretaria do campus "
    "Timóteo ou verificar o site oficial do CEFET-MG.\n\n"
    "Posso ajudar com mais alguma coisa?"
)

LOW_CONFIDENCE_DISCLAIMER = (
    "⚠️ *Com base nas informações disponíveis, esta é minha melhor resposta, "
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
    # Format retrieved chunks
    if retrieved_chunks:
        chunks_text = "\n\n---\n\n".join(
            f"**Fonte: {chunk.get('source', 'Documento institucional')}**\n{chunk['content']}"
            for chunk in retrieved_chunks
        )
    else:
        chunks_text = "(Nenhum documento relevante encontrado)"

    # Format conversation history
    history_text = "(Início da conversa)"
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:  # Last 6 turns max
            role = "Usuário" if turn["role"] == "user" else "Wander Jr"
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    # Build user message with context
    user_content = CONTEXT_TEMPLATE.format(
        retrieved_chunks=chunks_text,
        conversation_history=history_text,
        user_question=user_question,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
