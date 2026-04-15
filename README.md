# Wander Jr — Assistente Inteligente Institucional

Assistente inteligente do **CEFET-MG campus Timóteo** baseado em IA (LLM + RAG), capaz de responder dúvidas de alunos usando documentos oficiais da instituição.

- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Vector DB**: Qdrant (self-hosted)
- **Database**: PostgreSQL 16
- **Framework**: FastAPI + python-telegram-bot
- **Orquestração**: Implementação custom 

### 1. Clone e configure

```bash
cp .env.example .env
# Edite .env com suas chaves (TELEGRAM_BOT_TOKEN, GEMINI_API_KEY)
```

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
`EMBEDDING_REQUESTS_PER_MINUTE`, `EMBEDDING_MAX_RETRIES`, `RAG_EMBEDDING_BATCH_SIZE`.

### 4. Configure o webhook do Telegram

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
  -d "url=https://seu-dominio.com/webhook/telegram"
```

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
