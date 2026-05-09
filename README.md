# Wander Jr — Assistente Inteligente Institucional

O Wander Jr é um assistente inteligente desenvolvido para o CEFET-MG campus Timóteo, projetado para responder dúvidas com base nos documentos oficiais da instituição.

## Tecnologias usadas

- FastAPI
- Google Gemini
- Qdrant
- PostgreSQL
- python-telegram-bot

## Como rodar com Docker

1. Copie o arquivo de ambiente:

```bash
cp .env.example .env
```

2. Configure no `.env` as chaves necessárias, como `TELEGRAM_BOT_TOKEN` e `GEMINI_API_KEY`.

3. Suba o projeto:

```bash
docker compose up -d
```

4. Se precisar parar tudo:

```bash
docker compose down
```

## Telegram

Se você for usar o bot no Telegram, também é necessário configurar o webhook para apontar para o endereço público da aplicação.

Exemplo:

```bash
curl -X POST "https://api.telegram.org/bot<TOKEN>/setWebhook" \
	-d "url=https://seu-dominio.com/webhook/telegram"
```

## Licença

MIT
