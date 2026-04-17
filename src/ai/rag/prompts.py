"""Prompt templates for the Wander Jr RAG pipeline."""

from src.config.settings import (
    DEFAULT_APP_ASSISTANT_NAME,
    DEFAULT_APP_INSTITUTION_NAME,
    DEFAULT_RAG_PROMPT_HISTORY_TURNS,
)

SYSTEM_PROMPT_TEMPLATE = """\
Você é o {assistant_name}, o assistente virtual institucional do {institution_name}.
Sua missão é ajudar os alunos tirando dúvidas de forma educada, direta e acessível.

REGRAS OBRIGATÓRIAS:
1. Conteúdo factual deve ser 100% baseado nas informações institucionais fornecidas. Nunca invente informações.
2. Se a resposta não estiver claramente nas informações disponíveis, diga de forma simples que você não encontrou essa informação e não complete lacunas com suposições.
3. Nunca use termos técnicos como "contexto", "base vetorial", "documentos recuperados" ou expressões parecidas ao falar com o usuário.
4. Quando faltarem informações, responda de forma natural para um aluno leigo e oriente consulta ao setor competente, como secretaria, coordenação ou site oficial.
5. Para dúvidas sobre dados sensíveis ou acadêmicos (faltas, notas, histórico), informe que não possui acesso e oriente consulta no portal oficial (SIGAA).
6. Estilo pode ser flexível: linguagem natural, cordial e fácil de ler, sem alterar os fatos disponíveis.
7. Evite linguagem robótica ou excessivamente técnica. Soe como um atendente institucional claro e acolhedor.
8. Se houver pergunta factual junto com saudação/gíria (ex: "mas e aí?"), priorize responder a pergunta factual.
9. Se a mensagem for apenas saudação (ex: "Oi", "Bom dia", "Tudo bem?"), apenas cumprimente e pergunte como pode ajudar.
10. Se a pergunta pedir lista de pessoas/itens (ex: "quais são os professores"), liste todos os nomes/itens encontrados. Se a lista puder estar incompleta, avise de forma natural que pode não estar completa.
11. Evite respostas genéricas como "Como posso ajudar?" quando o usuário já fez uma pergunta objetiva.
12. Sempre que responder com informação factual, cite a fonte de forma breve e natural (ex: "Segundo o PPC..." ou "Conforme o guia da graduação...").
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    assistant_name=DEFAULT_APP_ASSISTANT_NAME,
    institution_name=DEFAULT_APP_INSTITUTION_NAME,
)

CONTEXT_TEMPLATE = """\
## INFORMAÇÕES INSTITUCIONAIS RELEVANTES:

{retrieved_chunks}

## INSTRUÇÃO DE RESPOSTA:
Use somente as informações acima para fatos. Você pode variar o tom e a organização do texto, mas sem adicionar fatos externos. Se a resposta não estiver claramente nas informações acima, diga apenas que não encontrou essa informação e sugira procurar o setor responsável, sem mencionar "contexto" ou termos técnicos.

## HISTÓRICO DA CONVERSA:
{conversation_history}

## PERGUNTA DO USUÁRIO:
{user_question}
"""

NO_CONTEXT_RESPONSE = (
    "Não encontrei essa informação no momento.\n\n"
    "Para confirmar direitinho, vale procurar a secretaria do campus Timóteo, a coordenação do curso ou o site oficial do CEFET-MG."
)

LOW_CONFIDENCE_DISCLAIMER = (
    "*Posso te orientar com o que encontrei, mas vale confirmar com o setor responsável:*\n\n"
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
    *,
    assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
    institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
    max_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
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
        for turn in conversation_history[-max_history_turns:]:
            role = "Usuário" if turn["role"] == "user" else assistant_name
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)

    user_content = CONTEXT_TEMPLATE.format(
        retrieved_chunks=chunks_text,
        conversation_history=history_text,
        user_question=user_question,
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                assistant_name=assistant_name,
                institution_name=institution_name,
            ),
        },
        {"role": "user", "content": user_content},
    ]
