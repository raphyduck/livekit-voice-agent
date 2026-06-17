# CLAUDE.md - livekit-voice-agent

## Contexte

Agent vocal conversationnel bidirectionnel pour Raphaël.
Répond au téléphone, parle français, a accès aux outils (agenda, email, tâches, Notion).

Stack :
- **Framework** : LiveKit Agents 1.5+ (Python)
- **STT** : Deepgram Nova-3 (fr-FR, streaming temps réel)
- **LLM** : Claude Sonnet 4.6 via livekit-plugins-anthropic
- **TTS** : Cartesia Sonic (voix FR, latence < 100ms)
- **Turn detection** : silero VAD + turn-detector sémantique
- **Outils** : MCP HTTP + APIs directes

## Structure

```
livekit-voice-agent/
├── CLAUDE.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── src/
    ├── agent.py           ← entrypoint LiveKit
    ├── tools.py           ← définition de tous les outils
    ├── mcp_client.py      ← client HTTP générique MCP
    └── system_prompt.py   ← prompt système
```

## requirements.txt

```
livekit-agents[anthropic,deepgram,cartesia,silero,turn-detector]>=1.5.0
python-dotenv>=1.0.0
httpx>=0.27.0
google-auth>=2.29.0
google-auth-httplib2>=0.2.0
google-api-python-client>=2.0.0
```

## Variables d'environnement (.env.example)

```bash
# LiveKit
LIVEKIT_URL=wss://xxx.livekit.cloud
LIVEKIT_API_KEY=APIxxx
LIVEKIT_API_SECRET=xxx

# STT
DEEPGRAM_API_KEY=xxx

# TTS
CARTESIA_API_KEY=xxx
CARTESIA_VOICE_ID=xxx   # choisir une voix FR depuis le dashboard Cartesia

# LLM
ANTHROPIC_API_KEY=sk-ant-xxx

# MCP self-hosted (Bearer tokens — obtenus depuis mcp-oauth-proxy)
IMAP_MCP_URL=https://imapmcp.hobbitton.at/mcp
IMAP_MCP_TOKEN=xxx

TWILIO_MCP_URL=https://twiliomcp.hobbitton.at/mcp
TWILIO_MCP_TOKEN=xxx

# Notion (Integration token — créer sur https://www.notion.so/my-integrations)
NOTION_API_KEY=secret_xxx
NOTION_BRAIN_PAGE_ID=381b975f-3b22-8070-924d-c105a4756e7a

# Todoist (token depuis https://todoist.com/prefs/integrations)
TODOIST_API_TOKEN=xxx

# Google (fichier credentials OAuth2 téléchargé depuis Google Cloud Console)
GOOGLE_CREDENTIALS_FILE=/app/google_credentials.json
GOOGLE_TOKEN_FILE=/app/google_token.json

# Agent
AGENT_NAME=Claude
AGENT_OWNER=Raphaël
```

---

## src/system_prompt.py

```python
SYSTEM_PROMPT = """
Tu es Claude, l'assistant vocal personnel de Raphaël Mainguy.
Tu réponds au téléphone en français.

RÈGLES ABSOLUES POUR LA VOIX :
- Réponses courtes : 1 à 3 phrases MAXIMUM par tour
- Aucun markdown, aucune liste, aucun titre, aucune puce
- Langage naturel et conversationnel, comme au téléphone
- Si tu dois réfléchir ou chercher une info, dis "un instant" avant d'utiliser un outil
- Une seule question à la fois si tu as besoin de précisions
- Pour les nombres, épelle-les naturellement en français

OUTILS DISPONIBLES :
- Agenda Google Calendar : voir et créer des événements
- Email : lire les derniers emails importants, envoyer
- Tâches Todoist : voir les tâches du jour, en ajouter
- Notion : consulter le brain de Raphaël
- SMS : envoyer un SMS depuis le numéro Twilio de Raphaël

COMPORTEMENT :
- Si quelqu'un appelle et que ce n'est pas Raphaël, prends un message
- Sois proactif : si l'heure est proche d'un RDV, mentionne-le
- En cas d'erreur d'outil, dis-le simplement et propose une alternative
"""
```

---

## src/mcp_client.py

Client HTTP générique pour appeler n'importe quel serveur MCP via HTTP streaming.

```python
import os
import httpx
import json
from typing import Any

class MCPClient:
    """Client HTTP pour serveurs MCP avec authentification Bearer."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def list_tools(self) -> list[dict]:
        """Retourne la liste des outils disponibles sur ce serveur MCP."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}",
                headers=self.headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=10.0
            )
            r.raise_for_status()
            data = r.json()
            return data.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Appelle un outil MCP et retourne le résultat."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{self.base_url}",
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments}
                },
                timeout=30.0
            )
            r.raise_for_status()
            data = r.json()
            result = data.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", str(result))
            return str(result)
```

---

## src/tools.py

Définir tous les outils disponibles pour l'agent.
Utiliser le décorateur `@llm.ai_callable()` de LiveKit pour chaque outil.

### Structure générale

```python
import os
import json
import httpx
from datetime import datetime
from livekit.agents.llm import ai_callable
from .mcp_client import MCPClient

# Clients MCP
_imap_client = MCPClient(os.environ["IMAP_MCP_URL"], os.environ["IMAP_MCP_TOKEN"])
_twilio_client = MCPClient(os.environ["TWILIO_MCP_URL"], os.environ["TWILIO_MCP_TOKEN"])
```

### Outils Google Calendar (via API directe)

```python
@ai_callable(description="Récupère les événements du calendrier Google pour aujourd'hui ou les prochains jours")
async def get_calendar_events(days_ahead: int = 1) -> str:
    """Retourne une liste lisible des événements à venir."""
    # Utiliser googleapiclient.discovery pour accéder à Calendar API v3
    # Lire GOOGLE_CREDENTIALS_FILE et GOOGLE_TOKEN_FILE
    # Appeler calendar.events().list() avec timeMin=now, timeMax=now+days_ahead
    # Retourner une string lisible par l'agent
    ...

@ai_callable(description="Crée un événement dans le calendrier Google")
async def create_calendar_event(
    title: str,
    start_datetime: str,  # ISO 8601
    end_datetime: str,    # ISO 8601
    description: str = ""
) -> str:
    ...
```

### Outils Email/IMAP (via MCP self-hosted)

```python
@ai_callable(description="Récupère les derniers emails non lus importants")
async def get_unread_emails(limit: int = 5) -> str:
    result = await _imap_client.call_tool("imap_get_latest_emails", {
        "folder": "INBOX",
        "limit": limit
    })
    return result

@ai_callable(description="Envoie un email")
async def send_email(to: str, subject: str, body: str) -> str:
    result = await _imap_client.call_tool("imap_send_email", {
        "to": to,
        "subject": subject,
        "body": body
    })
    return result
```

### Outils Todoist (via API directe)

```python
@ai_callable(description="Récupère les tâches Todoist du jour")
async def get_today_tasks() -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.todoist.com/rest/v2/tasks",
            headers={"Authorization": f"Bearer {os.environ['TODOIST_API_TOKEN']}"},
            params={"filter": "today | overdue"}
        )
        tasks = r.json()
        if not tasks:
            return "Aucune tâche pour aujourd'hui."
        return "\n".join([f"- {t['content']}" for t in tasks[:10]])

@ai_callable(description="Ajoute une tâche dans Todoist")
async def add_task(content: str, due_string: str = "aujourd'hui") -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.todoist.com/rest/v2/tasks",
            headers={
                "Authorization": f"Bearer {os.environ['TODOIST_API_TOKEN']}",
                "Content-Type": "application/json"
            },
            json={"content": content, "due_string": due_string, "due_lang": "fr"}
        )
        task = r.json()
        return f"Tâche ajoutée : {task.get('content')}"
```

### Outils Notion (via API directe)

```python
@ai_callable(description="Recherche dans le brain Notion de Raphaël")
async def search_notion(query: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.notion.com/v1/search",
            headers={
                "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            },
            json={"query": query, "page_size": 3}
        )
        results = r.json().get("results", [])
        if not results:
            return "Aucun résultat dans Notion."
        return "\n".join([r.get("properties", {}).get("title", {}).get("title", [{}])[0].get("plain_text", "?")
                          for r in results])
```

### Outil SMS Twilio (via MCP self-hosted)

```python
@ai_callable(description="Envoie un SMS depuis le numéro de Raphaël")
async def send_sms(to: str, message: str) -> str:
    result = await _twilio_client.call_tool("send_sms", {
        "to": to,
        "body": message
    })
    return result
```

### Outil heure courante

```python
@ai_callable(description="Retourne l'heure et la date actuelles")
async def get_current_datetime() -> str:
    now = datetime.now()
    return now.strftime("Nous sommes le %A %d %B %Y, il est %H heures %M.")
```

---

## src/agent.py

```python
import os
import asyncio
import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import anthropic, deepgram, cartesia, silero

from .system_prompt import SYSTEM_PROMPT
from .tools import (
    get_calendar_events, create_calendar_event,
    get_unread_emails, send_email,
    get_today_tasks, add_task,
    search_notion,
    send_sms,
    get_current_datetime,
)

load_dotenv()
logger = logging.getLogger("voice-agent")

async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Agent démarré pour la room: {ctx.room.name}")
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="fr",
            smart_format=True,
            punctuate=True,
        ),
        llm=anthropic.LLM(
            model="claude-sonnet-4-6",
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            model="sonic-multilingual",
            voice=os.environ["CARTESIA_VOICE_ID"],
            language="fr",
        ),
        vad=silero.VAD.load(),
        turn_detection="semantic",
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=SYSTEM_PROMPT,
            tools=[
                get_calendar_events,
                create_calendar_event,
                get_unread_emails,
                send_email,
                get_today_tasks,
                add_task,
                search_notion,
                send_sms,
                get_current_datetime,
            ],
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=True,
        ),
    )

    await session.generate_reply(
        instructions="Salue l'appelant en français, présente-toi comme Claude "
                     "l'assistant de Raphaël, et demande ce que tu peux faire."
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .env

CMD ["python", "-m", "src.agent", "start"]
```

## docker-compose.yml

```yaml
services:
  voice-agent:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./google_credentials.json:/app/google_credentials.json:ro
      - ./google_token.json:/app/google_token.json
    network_mode: host
```

---

## Contraintes

- Toujours utiliser `async/await` (LiveKit est entièrement async)
- Les outils doivent retourner des strings courtes et lisibles à voix haute
- Pas de markdown dans les retours d'outils (l'agent en fera du TTS)
- Logger sur stderr avec `logging`, jamais print()
- Les erreurs d'outils doivent être catchées et retourner un message lisible
- Garder les timeouts courts (5s pour lecture, 15s pour écriture)

## Après implémentation

```bash
git add -A && git commit -m "feat: livekit voice agent" && git push
```
