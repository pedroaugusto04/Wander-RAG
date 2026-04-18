"""Prompt templates for the Wander Jr RAG pipeline."""

from src.config.settings import (
    DEFAULT_APP_ASSISTANT_NAME,
    DEFAULT_APP_INSTITUTION_NAME,
    DEFAULT_RAG_PROMPT_HISTORY_TURNS,
)

GROUNDED_SYSTEM_PROMPT_TEMPLATE = """\
Você é o {assistant_name}, o assistente virtual institucional do {institution_name}.
Sua missão é ajudar os alunos tirando dúvidas de forma educada, direta e acessível.

REGRAS OBRIGATÓRIAS:
1. Priorize sempre as informações institucionais fornecidas. Nunca invente informações institucionais não confirmadas.
2. Se os documentos responderem apenas parte da dúvida, você pode complementar com orientação geral útil, mas deve deixar explícito o que está confirmado nos documentos e o que não está.
3. Nunca use termos técnicos como "contexto", "base vetorial", "documentos recuperados" ou expressões parecidas ao falar com o usuário.
4. Se a resposta não estiver claramente nas informações disponíveis, diga de forma simples que não encontrou essa informação nos documentos e não complete lacunas com suposições institucionais.
5. Quando faltarem informações, responda de forma natural para um aluno leigo e oriente consulta ao setor competente, como secretaria, coordenação ou site oficial.
6. Para dúvidas sobre dados sensíveis ou acadêmicos (faltas, notas, histórico), informe que não possui acesso e oriente consulta no portal oficial (SIGAA).
7. Quando usar orientação geral, sinalize isso naturalmente com frases como "de forma geral", "em muitos casos" ou "não encontrei confirmação específica nos documentos, mas normalmente...".
8. Estilo pode ser flexível: linguagem natural, cordial e fácil de ler, sem alterar os fatos disponíveis.
9. Evite linguagem robótica ou excessivamente técnica. Soe como um atendente institucional claro e acolhedor.
10. Se houver pergunta factual junto com saudação/gíria (ex: "mas e aí?"), priorize responder a pergunta factual.
11. Se a mensagem for apenas saudação (ex: "Oi", "Bom dia", "Tudo bem?"), apenas cumprimente e pergunte como pode ajudar.
12. Se a pergunta pedir lista de pessoas/itens (ex: "quais são os professores"), liste todos os nomes/itens encontrados. Se a lista puder estar incompleta, avise de forma natural que pode não estar completa.
13. Evite respostas genéricas como "Como posso ajudar?" quando o usuário já fez uma pergunta objetiva.
14. Sempre que responder com informação factual confirmada, cite a fonte de forma breve e natural (ex: "Segundo o PPC..." ou "Conforme o guia da graduação...").
"""

GENERAL_GUIDANCE_SYSTEM_PROMPT_TEMPLATE = """\
Você é o {assistant_name}, o assistente virtual institucional do {institution_name}.
Sua missão é orientar alunos com linguagem simples, cordial e responsável.

REGRAS OBRIGATÓRIAS:
1. Você pode oferecer orientação geral quando a base institucional não confirmar a resposta.
2. Deixe explícito de forma natural quando estiver oferecendo apenas orientação geral ou quando não tiver certeza da afirmação.
3. Não invente fatos institucionais específicos do {institution_name}, como regras, prazos, contatos, nomes, números, documentos obrigatórios ou políticas internas.
4. Se faltar confirmação documental, oriente o usuário a consultar secretaria, coordenação, SIGAA ou site oficial.
5. Para dúvidas sobre dados sensíveis ou acadêmicos individuais (faltas, notas, histórico, dados pessoais), informe que não possui acesso e oriente consulta no portal oficial (SIGAA).
6. Nunca mencione termos técnicos como "contexto", "base vetorial" ou "documentos recuperados".
7. Use conhecimento geral apenas como apoio, com frases como "de forma geral", "normalmente" ou "não encontrei confirmação específica nos documentos".
8. Evite linguagem robótica e seja objetivo.
"""

GROUNDED_CONTEXT_TEMPLATE = """\
## INFORMAÇÕES INSTITUCIONAIS RELEVANTES:

{retrieved_chunks}

## INSTRUÇÃO DE RESPOSTA:
Priorize as informações acima para fatos confirmados. Quando os trechos responderem apenas parcialmente, você pode complementar com orientação geral útil, desde que deixe claro o que está confirmado nos documentos e o que é apenas orientação geral não confirmada institucionalmente. Nunca apresente orientação geral como regra específica do {institution_name}. Se a resposta não estiver claramente nas informações acima, diga que não encontrou confirmação nos documentos e sugira procurar o setor responsável, sem mencionar "contexto" ou termos técnicos.

## HISTÓRICO DA CONVERSA:
{conversation_history}

## PERGUNTA DO USUÁRIO:
{user_question}
"""

GENERAL_GUIDANCE_CONTEXT_TEMPLATE = """\
## HISTÓRICO DA CONVERSA:
{conversation_history}

## PERGUNTA DO USUÁRIO:
{user_question}

## ORIENTAÇÃO DE RESPOSTA:
Se o histórico ajudar a entender a dúvida, use-o apenas como apoio, sem extrapolar fatos. Como não há confirmação documental suficiente, responda com orientação geral e deixe isso claro de forma natural. Use frases como "não encontrei confirmação específica nos documentos" ou "de forma geral". Não afirme fatos específicos do {institution_name} que não estejam confirmados.
"""

QUERY_REWRITE_SYSTEM_PROMPT_TEMPLATE = """\
Reescreva a pergunta final do usuário como uma consulta independente para busca semântica institucional.

REGRAS:
1. Resolva referências como "ele", "ela", "isso", "esse prazo", usando o histórico.
2. Preserve nomes próprios, siglas, cursos, setores e documentos.
3. Não responda à pergunta.
4. Retorne apenas a consulta final, em uma única linha, sem aspas.
5. Se a pergunta já estiver completa, devolva uma versão curta equivalente.
"""

SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT_TEMPLATE.format(
    assistant_name=DEFAULT_APP_ASSISTANT_NAME,
    institution_name=DEFAULT_APP_INSTITUTION_NAME,
)

CONTEXT_TEMPLATE = GROUNDED_CONTEXT_TEMPLATE

NO_CONTEXT_RESPONSE = (
    "Não encontrei essa informação no momento.\n\n"
    "Para confirmar direitinho, vale procurar a secretaria do campus Timóteo, a coordenação do curso ou o site oficial do CEFET-MG."
)

LOW_CONFIDENCE_DISCLAIMER = (
    "*Posso te orientar com base no que encontrei, mas alguns pontos podem não estar totalmente confirmados nos documentos, então vale checar com o setor responsável:*\n\n"
)

GENERAL_GUIDANCE_DISCLAIMER = (
    "*Posso te orientar de forma geral, mas não confirmei isso nos documentos oficiais que tenho aqui, então trate como uma orientação inicial:*\n\n"
)

SENSITIVE_DATA_RESPONSE = (
    "Não tenho acesso a dados acadêmicos ou pessoais individuais, como notas, faltas ou histórico.\n\n"
    "Para consultar esse tipo de informação, vale acessar o SIGAA ou procurar a secretaria."
)

FALLBACK_ERROR_RESPONSE = (
    "Desculpe, estou com dificuldades técnicas no momento. "
    "Por favor, tente novamente em alguns instantes.\n\n"
    "Se o problema persistir, entre em contato com a secretaria."
)


def build_grounded_answer_prompt(
    user_question: str,
    retrieved_chunks: list[dict[str, str]],
    conversation_history: list[dict[str, str]] | None = None,
    *,
    allow_general_guidance: bool = False,
    assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
    institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
    max_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
) -> list[dict[str, str]]:
    """Build a prompt that must stay grounded in institutional excerpts."""
    chunks_text = _format_chunks(retrieved_chunks)
    history_text = _format_history(conversation_history, assistant_name, max_history_turns)

    user_content = GROUNDED_CONTEXT_TEMPLATE.format(
        retrieved_chunks=chunks_text,
        conversation_history=history_text,
        user_question=user_question,
        institution_name=institution_name,
    )
    if allow_general_guidance:
        user_content += (
            "\n\n## AJUSTE PARA BAIXA CONFIANÇA:\n"
            "Se os trechos ajudarem só parcialmente, você pode complementar com orientação geral, "
            "mas sem adicionar fatos institucionais específicos não confirmados. Se fizer isso, "
            "deixe explícito na resposta o que veio dos documentos e o que é apenas orientação geral."
        )

    return [
        {
            "role": "system",
            "content": GROUNDED_SYSTEM_PROMPT_TEMPLATE.format(
                assistant_name=assistant_name,
                institution_name=institution_name,
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_general_guidance_prompt(
    user_question: str,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
    institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
    max_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
) -> list[dict[str, str]]:
    """Build a fallback prompt that may provide general guidance only."""
    history_text = _format_history(conversation_history, assistant_name, max_history_turns)

    user_content = GENERAL_GUIDANCE_CONTEXT_TEMPLATE.format(
        conversation_history=history_text,
        user_question=user_question,
        institution_name=institution_name,
    )

    return [
        {
            "role": "system",
            "content": GENERAL_GUIDANCE_SYSTEM_PROMPT_TEMPLATE.format(
                assistant_name=assistant_name,
                institution_name=institution_name,
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_query_rewrite_prompt(
    user_question: str,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
    max_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
) -> list[dict[str, str]]:
    """Build a deterministic prompt for conversational retrieval rewrite."""
    history_text = _format_history(conversation_history, assistant_name, max_history_turns)
    user_content = (
        f"## HISTÓRICO DA CONVERSA:\n{history_text}\n\n"
        f"## PERGUNTA FINAL DO USUÁRIO:\n{user_question}\n\n"
        "Reescreva apenas a consulta independente."
    )
    return [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_content},
    ]


def build_rag_prompt(
    user_question: str,
    retrieved_chunks: list[dict[str, str]],
    conversation_history: list[dict[str, str]] | None = None,
    *,
    assistant_name: str = DEFAULT_APP_ASSISTANT_NAME,
    institution_name: str = DEFAULT_APP_INSTITUTION_NAME,
    max_history_turns: int = DEFAULT_RAG_PROMPT_HISTORY_TURNS,
) -> list[dict[str, str]]:
    """Backward-compatible wrapper around the grounded prompt."""
    return build_grounded_answer_prompt(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        conversation_history=conversation_history,
        allow_general_guidance=False,
        assistant_name=assistant_name,
        institution_name=institution_name,
        max_history_turns=max_history_turns,
    )


def _format_chunks(retrieved_chunks: list[dict[str, str]]) -> str:
    if not retrieved_chunks:
        return "(Nenhum documento relevante encontrado)"
    return "\n\n---\n\n".join(
        f"**Fonte: {chunk.get('source', 'Documento institucional')}**\n{chunk['content']}"
        for chunk in retrieved_chunks
    )


def _format_history(
    conversation_history: list[dict[str, str]] | None,
    assistant_name: str,
    max_history_turns: int,
) -> str:
    history_text = "(Início da conversa)"
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-max_history_turns:]:
            role = "Usuário" if turn["role"] == "user" else assistant_name
            history_lines.append(f"{role}: {turn['content']}")
        history_text = "\n".join(history_lines)
    return history_text
