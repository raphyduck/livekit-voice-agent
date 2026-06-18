import os
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from livekit import api
from livekit.agents import function_tool, get_job_context

from .mcp_client import MCPClient

logger = logging.getLogger("voice-agent.tools")

# Timeouts courts : lecture 5s, écriture 15s
READ_TIMEOUT = 5.0
WRITE_TIMEOUT = 15.0

# Clients MCP self-hosted (instanciés paresseusement pour éviter une erreur
# si les variables d'environnement ne sont pas encore chargées à l'import).
_imap_client: MCPClient | None = None
_twilio_client: MCPClient | None = None


def _imap() -> MCPClient:
    global _imap_client
    if _imap_client is None:
        _imap_client = MCPClient(os.environ["IMAP_MCP_URL"], os.environ["IMAP_MCP_TOKEN"])
    return _imap_client


def _twilio() -> MCPClient:
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = MCPClient(os.environ["TWILIO_MCP_URL"], os.environ["TWILIO_MCP_TOKEN"])
    return _twilio_client


# ---------------------------------------------------------------------------
# Google Calendar (API directe)
# ---------------------------------------------------------------------------

_GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _calendar_service():
    """Construit un service Calendar v3 authentifié à partir des fichiers OAuth2.

    Rafraîchit le token si nécessaire et le réécrit sur disque.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_file = os.environ["GOOGLE_TOKEN_FILE"]
    creds = Credentials.from_authorized_user_file(token_file, _GOOGLE_SCOPES)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _format_event(event: dict) -> str:
    start = event.get("start", {})
    start_str = start.get("dateTime") or start.get("date", "")
    summary = event.get("summary", "Sans titre")
    try:
        dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        when = dt.strftime("%d/%m à %Hh%M") if "T" in start_str else dt.strftime("%d/%m")
    except ValueError:
        when = start_str
    return f"{when} : {summary}"


@function_tool()
async def get_calendar_events(days_ahead: int = 1) -> str:
    """Récupère les événements du calendrier Google pour aujourd'hui ou les prochains jours.

    Args:
        days_ahead: Nombre de jours à regarder à partir de maintenant (1 = aujourd'hui).
    """
    try:
        service = _calendar_service()
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=max(1, days_ahead))).isoformat()

        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=10,
        ).execute()

        events = result.get("items", [])
        if not events:
            return "Aucun événement prévu sur cette période."
        return ". ".join(_format_event(e) for e in events)
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur get_calendar_events")
        return f"Je n'ai pas pu consulter l'agenda : {e}"


@function_tool()
async def create_calendar_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
) -> str:
    """Crée un événement dans le calendrier Google.

    Args:
        title: Titre de l'événement.
        start_datetime: Début au format ISO 8601 (ex: 2026-06-17T14:00:00).
        end_datetime: Fin au format ISO 8601.
        description: Description optionnelle.
    """
    try:
        service = _calendar_service()
        body = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_datetime, "timeZone": "Europe/Paris"},
            "end": {"dateTime": end_datetime, "timeZone": "Europe/Paris"},
        }
        service.events().insert(calendarId="primary", body=body).execute()
        return f"C'est noté, j'ai créé l'événement {title}."
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur create_calendar_event")
        return f"Je n'ai pas pu créer l'événement : {e}"


# ---------------------------------------------------------------------------
# Email / IMAP (MCP self-hosted)
# ---------------------------------------------------------------------------

@function_tool()
async def get_unread_emails(limit: int = 5) -> str:
    """Récupère les derniers emails importants de la boîte de réception.

    Args:
        limit: Nombre maximum d'emails à retourner.
    """
    try:
        return await _imap().call_tool(
            "imap_get_latest_emails", {"folder": "INBOX", "limit": limit}
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur get_unread_emails")
        return f"Je n'ai pas pu lire les emails : {e}"


@function_tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """Envoie un email.

    Args:
        to: Adresse du destinataire.
        subject: Objet de l'email.
        body: Contenu de l'email.
    """
    try:
        return await _imap().call_tool(
            "imap_send_email", {"to": to, "subject": subject, "body": body}
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur send_email")
        return f"Je n'ai pas pu envoyer l'email : {e}"


# ---------------------------------------------------------------------------
# Todoist (API directe)
# ---------------------------------------------------------------------------

@function_tool()
async def get_today_tasks() -> str:
    """Récupère les tâches Todoist du jour et en retard."""
    try:
        async with httpx.AsyncClient(timeout=READ_TIMEOUT) as client:
            r = await client.get(
                "https://api.todoist.com/rest/v2/tasks",
                headers={"Authorization": f"Bearer {os.environ['TODOIST_API_TOKEN']}"},
                params={"filter": "today | overdue"},
            )
            r.raise_for_status()
            tasks = r.json()
        if not tasks:
            return "Aucune tâche pour aujourd'hui."
        return ". ".join(t["content"] for t in tasks[:10])
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur get_today_tasks")
        return f"Je n'ai pas pu consulter les tâches : {e}"


@function_tool()
async def add_task(content: str, due_string: str = "aujourd'hui") -> str:
    """Ajoute une tâche dans Todoist.

    Args:
        content: Intitulé de la tâche.
        due_string: Échéance en langage naturel (ex: aujourd'hui, demain, lundi).
    """
    try:
        async with httpx.AsyncClient(timeout=WRITE_TIMEOUT) as client:
            r = await client.post(
                "https://api.todoist.com/rest/v2/tasks",
                headers={
                    "Authorization": f"Bearer {os.environ['TODOIST_API_TOKEN']}",
                    "Content-Type": "application/json",
                },
                json={"content": content, "due_string": due_string, "due_lang": "fr"},
            )
            r.raise_for_status()
            task = r.json()
        return f"Tâche ajoutée : {task.get('content')}"
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur add_task")
        return f"Je n'ai pas pu ajouter la tâche : {e}"


# ---------------------------------------------------------------------------
# Notion (API directe)
# ---------------------------------------------------------------------------

def _extract_notion_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_parts = prop.get("title", [])
            if title_parts:
                return title_parts[0].get("plain_text", "?")
    # Fallback pour les résultats de type "database" ou titre au niveau racine.
    title_parts = page.get("title", [])
    if title_parts:
        return title_parts[0].get("plain_text", "?")
    return "Sans titre"


@function_tool()
async def search_notion(query: str) -> str:
    """Recherche dans le brain Notion de Raphaël.

    Args:
        query: Termes à rechercher.
    """
    try:
        async with httpx.AsyncClient(timeout=READ_TIMEOUT) as client:
            r = await client.post(
                "https://api.notion.com/v1/search",
                headers={
                    "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json",
                },
                json={"query": query, "page_size": 3},
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        if not results:
            return "Aucun résultat dans Notion."
        return ". ".join(_extract_notion_title(p) for p in results)
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur search_notion")
        return f"Je n'ai pas pu chercher dans Notion : {e}"


# ---------------------------------------------------------------------------
# SMS Twilio (MCP self-hosted)
# ---------------------------------------------------------------------------

@function_tool()
async def send_sms(to: str, message: str) -> str:
    """Envoie un SMS depuis le numéro de Raphaël.

    Args:
        to: Numéro de téléphone du destinataire.
        message: Contenu du SMS.
    """
    try:
        return await _twilio().call_tool("send_sms", {"to": to, "body": message})
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur send_sms")
        return f"Je n'ai pas pu envoyer le SMS : {e}"


# ---------------------------------------------------------------------------
# Heure courante
# ---------------------------------------------------------------------------

_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


@function_tool()
async def get_current_datetime() -> str:
    """Retourne l'heure et la date actuelles."""
    now = datetime.now()
    jour = _JOURS[now.weekday()]
    mois = _MOIS[now.month - 1]
    return (
        f"Nous sommes le {jour} {now.day} {mois} {now.year}, "
        f"il est {now.hour} heures {now.minute:02d}."
    )


# ---------------------------------------------------------------------------
# Raccrochage
# ---------------------------------------------------------------------------

@function_tool()
async def end_call() -> str:
    """Raccroche et met fin à l'appel téléphonique en cours.

    À utiliser quand l'interlocuteur dit au revoir, que la conversation est
    clairement terminée, ou qu'il n'y a plus rien à faire.
    """
    try:
        ctx = get_job_context()
        # Attendre que l'agent ait FINI de parler (TTS drainé) avant de couper.
        session = getattr(ctx, "primary_session", None)
        if session is not None:
            try:
                await session.wait_for_idle()
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2.0)  # fallback si wait_for_idle indispo
        else:
            await asyncio.sleep(2.0)
        # petite marge pour la latence réseau du dernier paquet audio
        await asyncio.sleep(0.3)
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        return "Appel terminé."
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur end_call")
        return f"Je n'ai pas pu raccrocher : {e}"
