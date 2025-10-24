# Deploying the Enhanced Telegram DS Bot to Railway

1. Create a new Railway project and connect your GitHub repo or upload the code.
2. Add environment variables in Railway:
   - TELEGRAM_TOKEN
   - OPENAI_API_KEY (optional)
   - ENABLE_AUTOML (true/false)
3. Railway will build the Dockerfile included. The container runs `python main.py`.
4. For production, prefer webhook mode (modify main.py) and configure Railway to expose a public URL and set the webhook.
5. Monitor logs, and consider using a worker queue (Redis + RQ) if you process large datasets frequently.
