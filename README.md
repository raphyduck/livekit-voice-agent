# livekit-voice-agent

Agent vocal conversationnel bidirectionnel en français pour Raphaël. Répond au
téléphone, parle français, et a accès à des outils (agenda, email, tâches, Notion, SMS).

## Stack

- **Framework** : [LiveKit Agents](https://docs.livekit.io/agents/) 1.5+ (Python)
- **STT** : Deepgram Nova-3 (fr-FR, streaming temps réel)
- **LLM** : Claude Sonnet 4.6 via `livekit-plugins-anthropic`
- **TTS** : Cartesia Sonic (voix FR)
- **Turn detection** : silero VAD + turn-detector sémantique multilingue
- **Outils** : MCP HTTP self-hosted + APIs directes

## Structure

```
livekit-voice-agent/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── src/
    ├── agent.py          ← entrypoint LiveKit
    ├── tools.py          ← définition de tous les outils
    ├── mcp_client.py     ← client HTTP générique MCP
    └── system_prompt.py  ← prompt système
```

## Configuration

1. Copier `.env.example` vers `.env` et renseigner les clés.
2. Pour Google Calendar : déposer `google_credentials.json` (OAuth2 desktop)
   et générer `google_token.json` (premier flow OAuth) à la racine.

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.agent download-files   # pré-télécharge VAD + turn-detector
python -m src.agent dev              # mode développement
```

## Lancer avec Docker

```bash
docker compose up --build -d
docker compose logs -f
```

## Outils disponibles

| Outil | Intégration |
| --- | --- |
| `get_calendar_events`, `create_calendar_event` | Google Calendar (API directe) |
| `get_unread_emails`, `send_email` | IMAP MCP self-hosted |
| `get_today_tasks`, `add_task` | Todoist (API directe) |
| `search_notion` | Notion (API directe) |
| `send_sms` | Twilio MCP self-hosted |
| `get_current_datetime` | local |
