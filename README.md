# Wander Jr — Assistente Inteligente Institucional

Assistente inteligente do **CEFET-MG campus Timóteo** baseado em IA (LLM + RAG), capaz de responder dúvidas de alunos usando documentos oficiais da instituição.

- **LLM**: Google Gemini 2.5 Flash
- **Vector DB**: Qdrant (self-hosted)
- **Database**: PostgreSQL 16
- **Framework**: FastAPI + python-telegram-bot
- **Orquestração**: Implementação custom 

### 1. Clone e configure

```bash
cp .env.example .env
# Edite .env com suas chaves (TELEGRAM_BOT_TOKEN, GEMINI_API_KEY)
# O projeto também aceita `.env.local` para sobrescritas locais não versionadas
# Opcional: configure fallback automático de modelos
# LLM_FALLBACK_MODELS=gemini-3.1-flash-lite,gemma-3-12b
# EMBEDDING_FALLBACK_MODELS=models/gemini-embedding-001
```

Algumas observações úteis:
- `DATABASE_URL` é opcional. Se ficar vazio, ela é montada automaticamente a partir de `POSTGRES_*`.
- As respostas institucionais fixas e metadados do app agora podem ser ajustados por `APP_*`.
- Configurações de retrieval e ingestão ficam concentradas em `RAG_*`.
- Configurações de Docker/Compose ficam no final do `.env.example`.

### 2. Suba os serviços

```bash
docker compose up -d
```

### 3. Ingira documentos

```bash
# Coloque PDFs em data/documents/
# A ingestão é manual (não é executada automaticamente no startup).

# (Opcional) Forçar ingestão manual
docker compose exec app python scripts/ingest_documents.py data/documents/

# Ver info da collection
docker compose exec app python scripts/ingest_documents.py --info .
```

Se estiver batendo quota de embedding no Gemini, ajuste no `.env`:
`EMBEDDING_REQUESTS_PER_MINUTE`, `EMBEDDING_MAX_RETRIES`,
`EMBEDDING_BASE_RETRY_SECONDS`, `RAG_EMBEDDING_BATCH_SIZE`.

Para fallback automático de chat e embedding, ajuste também:
`LLM_FALLBACK_MODELS` e `EMBEDDING_FALLBACK_MODELS`.

Para ligar/desligar reranker no `.env`:
`RERANKER_ENABLED=true` (ligado) ou `RERANKER_ENABLED=false` (desligado).

Para perguntas em formato de lista e histórico do prompt, ajuste:
`RAG_LIST_QUERY_MIN_TOP_K` e `RAG_PROMPT_HISTORY_TURNS`.

### 4. Configure o webhook do Telegram

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -d "url=https://seu-dominio.com/webhook/telegram"
```

## Variáveis no GitHub / Deploy

- Nunca suba o arquivo `.env` no repositório.
- Suba apenas o `.env.example` com placeholders.
- No ambiente de deploy (Render, Railway, VPS, etc.), configure as variáveis de ambiente diretamente no painel.
- Se usar GitHub Actions, configure chaves sensíveis em `Settings > Secrets and variables > Actions`.
- Para controlar reranker em produção, defina `RERANKER_ENABLED=true|false` nesse ambiente.

## Estrutura do Projeto

```
src/
├── main.py                  # FastAPI entry point
├── config/settings.py       # Configuração (env vars)
├── channels/                # Abstração de canais
│   ├── base.py              # Interface MessageChannelAdapter
│   └── telegram/            # Implementação Telegram
├── core/                    # Lógica de aplicação
│   ├── models.py            # Domain models
│   ├── orchestrator.py      # AI Orchestrator
│   └── conversation.py      # Session manager
├── ai/                      # Camada de IA
│   ├── llm/                 # LLM providers (Gemini)
│   └── rag/                 # RAG pipeline + prompts
├── knowledge/               # Base de conhecimento
│   ├── ingestion/           # Loaders, chunking, pipeline
│   └── vectorstore/         # Qdrant implementation
└── infra/                   # Logging, database, cache
```

## 🛠 Desenvolvimento Local

```bash
# Instalar dependências
uv pip install -e ".[dev]"

# Rodar testes
pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## 📄 Licença

MIT
