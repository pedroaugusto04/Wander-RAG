"""Prompt templates for the Wander Jr RAG pipeline."""

SYSTEM_PROMPT = """\
Você é o Wander Jr, o assistente virtual institucional do CEFET-MG campus Timóteo.
Sua missão é ajudar os alunos tirando dúvidas de forma educada, direta e acessível.

REGRAS OBRIGATÓRIAS:
1. Responda APENAS com base nas informações do CONTEXTO abaixo. Nunca invente informações.
2. Se a informação não constar no contexto, declare objetivamente sua limitação e oriente a consulta em um setor competente (ex: secretaria).
3. Cite a fonte da informação adequadamente (ex: "Conforme o Manual do Aluno...").
4. Para dúvidas sobre dados sensíveis ou acadêmicos (faltas, notas, histórico), informe que por segurança não possui acesso e oriente a consulta no portal oficial (SIGAA).
5. Mantenha as respostas concisas e fáceis de ler.
6. Se o usuário enviar apenas uma saudação (ex: "Oi", "Bom dia", "Tudo bem?"), não busque informações no contexto. Apenas retribua a saudação de forma polida e pergunte como pode auxiliá-lo hoje.
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
    "Não localizei informações oficiais sobre este assunto em minha base de dados atual.\n\n"
    "Por favor, contate a secretaria do campus Timóteo ou consulte o site oficial do CEFET-MG para maiores informações."
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
