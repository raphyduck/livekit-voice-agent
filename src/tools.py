import difflib
import json
import os
import unicodedata
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from livekit import api
from livekit.agents import function_tool, get_job_context

from .mcp_client import MCPClient

logger = logging.getLogger("voice-agent.tools")

# Timeouts courts : lecture 5s, écriture 15s
READ_TIMEOUT = 5.0
WRITE_TIMEOUT = 15.0

# ---------------------------------------------------------------------------
# Identité de l'appelant (authentification faible : caller ID + mot de passe)
# ---------------------------------------------------------------------------
# Le worker traite un appel par process : une variable de module est donc
# acceptable ici. On la remet à False au début de chaque appel (reset_identity).
_IDENTITY = {"is_raphael": False, "mdp_verifie": False}

# Message neutre renvoyé par les outils sensibles tant que l'identité n'est pas
# confirmée (ne révèle rien sur l'existence d'un mot de passe).
_ACCESS_DENIED = (
    "Désolé, cette information est réservée à Raphaël. Je peux seulement prendre "
    "un message ou donner des informations générales."
)


def set_direction_sortante(sortant: bool) -> None:
    """Memorise le sens de l'appel : la passerelle en depend (voir _palier_courant)."""
    _IDENTITY["sortant"] = bool(sortant)


def est_sortant() -> bool:
    return bool(_IDENTITY.get("sortant"))


def reset_identity(value: bool = False) -> None:
    # is_raphael : presomption (caller ID), falsifiable.
    # mdp_verifie : preuve (mot de passe vocal), le seul fait qui ouvre le
    #   palier 2 de la passerelle. Remis a zero a chaque appel, jamais deduit
    #   du caller ID — sinon usurper un numero suffirait a lire les comptes.
    _IDENTITY["is_raphael"] = value
    _IDENTITY["mdp_verifie"] = False
    _IDENTITY["sortant"] = False
    _IDENTITY["essais_mdp"] = 0


def is_raphael() -> bool:
    return _IDENTITY["is_raphael"]


def mdp_verifie() -> bool:
    return bool(_IDENTITY.get("mdp_verifie"))

# Etat par appel : un compte-rendu a-t-il deja ete ecrit ?
# journal_page_id : id de la page Journal creee pour CET appel (idempotence).
# journal_detail  : texte accumule pendant l'appel, ecrit en une seule entree
#                   a la fin (le journal du cerveau est append-only).
# journal_title   : titre de l'entree unique de cet appel.
_CALL_STATE = {"journal_written": False, "journal_page_id": None,
               "journal_detail": "", "journal_title": "", "journal_type": "info"}


def reset_call_state() -> None:
    _CALL_STATE["journal_written"] = False
    _CALL_STATE["journal_page_id"] = None
    _CALL_STATE["journal_detail"] = ""
    _CALL_STATE["journal_title"] = ""
    _CALL_STATE["journal_type"] = "info"


def journal_was_written() -> bool:
    return _CALL_STATE["journal_written"]


def journal_page_id():
    return _CALL_STATE["journal_page_id"]

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
    if not is_raphael():
        return _ACCESS_DENIED
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
    if not is_raphael():
        return _ACCESS_DENIED
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
    if not is_raphael():
        return _ACCESS_DENIED
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
    if not is_raphael():
        return _ACCESS_DENIED
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
# Cerveau hobbitton (lecture : montage RO /brain ; ecriture : MCP memoire)
# ---------------------------------------------------------------------------
# Historique : cette section tapait l'API Notion en direct. Depuis le
# 24/08/2026 la source de verite est le depot memoire (hobbitton-memory).
# Lecture = fichiers du montage /brain, sans reseau : c'est ce qui tient la
# latence d'un appel telephonique. Ecriture = outil `journaliser` du MCP
# memoire, qui produit le commit git signe et respecte l'append-only.

_BRAIN_DIR = os.environ.get("BRAIN_DIR", "/brain")
_MEMORY_MCP_URL = os.environ.get("MEMORY_MCP_URL", "http://127.0.0.1:8091/mcp")
_JOURNAL_TYPES = {"info", "action", "erreur", "décision requise"}


def _brain_path(*parts) -> str:
    return os.path.join(_BRAIN_DIR, *parts)


def _read_note(rel: str, limit: int = 4000) -> str:
    try:
        with open(_brain_path(rel), encoding="utf-8") as fh:
            return fh.read(limit)
    except Exception:  # noqa: BLE001
        return ""


async def _memory_call(tool: str, arguments: dict, timeout: float = WRITE_TIMEOUT) -> str:
    """Appelle un outil du MCP memoire. Renvoie le texte du resultat.

    Le transport rend du SSE : une ou plusieurs lignes `data: {json}`.
    """
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(_MEMORY_MCP_URL, headers=headers, json=payload)
        r.raise_for_status()
        body = r.text
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        import json as _json
        data = _json.loads(line[5:].strip())
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        content = data.get("result", {}).get("content") or []
        if content:
            return content[0].get("text", "")
        return ""
    raise RuntimeError("reponse MCP memoire illisible")


@function_tool()
async def read_brain(query: str) -> str:
    """Consulte le cerveau de Raphael : profil, entites, procedures, items ouverts.

    Renvoie le contenu des passages les plus pertinents, pas seulement les titres.

    Args:
        query: Termes a rechercher (sujet, nom d'entite, infra, etc.).
    """
    if not is_raphael():
        return _ACCESS_DENIED
    try:
        terms = [t.lower() for t in re.findall(r"\w{3,}", query or "")][:6]
        out = []
        index = _read_note("index.md", 3500)
        if index:
            out.append("INDEX DU CERVEAU\n" + index)
        if terms:
            hits = []
            for racine, dirs, fichiers in os.walk(_BRAIN_DIR):
                dirs[:] = [d for d in dirs if d not in (".git", "journal")]
                for f in fichiers:
                    if not f.endswith(".md"):
                        continue
                    chemin = os.path.join(racine, f)
                    rel = os.path.relpath(chemin, _BRAIN_DIR)
                    if rel == "index.md":
                        continue
                    try:
                        with open(chemin, encoding="utf-8") as fh:
                            texte = fh.read()
                    except Exception:  # noqa: BLE001
                        continue
                    bas = texte.lower()
                    score = sum(bas.count(t) for t in terms)
                    if score:
                        pos = min((bas.find(t) for t in terms if bas.find(t) >= 0),
                                  default=0)
                        debut = max(0, pos - 200)
                        hits.append((score, rel, texte[debut:debut + 1200]))
            hits.sort(key=lambda h: -h[0])
            for _score, rel, extrait in hits[:3]:
                out.append(rel + "\n" + extrait)
        if not out:
            return "Je n'ai rien trouve dans le cerveau sur ce sujet."
        return "\n\n---\n\n".join(out)[:6000]
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur read_brain")
        return f"Je n'ai pas pu consulter le cerveau : {e}"


async def flush_journal() -> bool:
    """Ecrit UNE entree de journal pour l'appel en cours, puis vide le tampon.

    Le journal du cerveau est append-only : on ne peut pas corriger une entree
    deja ecrite. On accumule donc tout pendant l'appel et on ecrit une seule
    fois, a la fin. Renvoie True si une entree a ete ecrite.
    """
    tampon = _CALL_STATE.get("journal_detail") or ""
    titre = _CALL_STATE.get("journal_title") or ""
    if not (tampon or titre):
        return False
    entree = titre
    if tampon:
        entree = (titre + " — " + tampon) if titre else tampon
    try:
        await _memory_call("journaliser", {
            "entree": entree[:12000],
            "auteur": "agent vocal (LiveKit)",
            "tags": ["appel", _CALL_STATE.get("journal_type") or "info"],
        })
        _CALL_STATE["journal_detail"] = ""
        _CALL_STATE["journal_title"] = ""
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Erreur flush_journal")
        return False


async def write_journal_raw(action: str, detail: str = "", type: str = "info") -> str:
    """Note une trace d'activite pour l'appel en cours.

    N'ecrit PAS tout de suite : le journal du cerveau est append-only, donc on
    accumule et on ecrit une entree unique en fin d'appel (flush_journal).

    Args:
        action: Resume court de ce qui s'est passe.
        detail: Details complementaires (optionnel).
        type: Categorie : 'info', 'action', 'erreur' ou 'decision requise'.
    """
    # NB : pas de garde is_raphael() ici. Le journal est une trace interne,
    # jamais lue a l'appelant, et doit rester disponible sur les appels sortants.
    if type not in _JOURNAL_TYPES:
        type = "info"
    if not _CALL_STATE.get("journal_title"):
        _CALL_STATE["journal_title"] = action[:200]
        _CALL_STATE["journal_type"] = type
    else:
        suite = (action + (" — " + detail if detail else ""))
        prev = _CALL_STATE.get("journal_detail") or ""
        _CALL_STATE["journal_detail"] = (prev + "\n— maj : " + suite)[:11000]
        _CALL_STATE["journal_written"] = True
        _CALL_STATE["journal_page_id"] = "buffer"
        return "Note ajoutee au journal de l'appel."
    if detail:
        _CALL_STATE["journal_detail"] = detail[:11000]
    _CALL_STATE["journal_written"] = True
    _CALL_STATE["journal_page_id"] = "buffer"
    return "Note ajoutee au journal de l'appel."


@function_tool()
async def write_journal(action: str, detail: str = "", type: str = "info") -> str:
    """Ajoute une entree au journal du cerveau (trace interne)."""
    return await write_journal_raw(action, detail, type)


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
    if not is_raphael():
        return _ACCESS_DENIED
    try:
        return await _twilio().call_tool("send_sms", {"to": to, "body": message})
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur send_sms")
        return f"Je n'ai pas pu envoyer le SMS : {e}"


@function_tool()
async def lire_sms(limit: int = 10) -> str:
    """Liste les SMS récents (reçus et envoyés) du numéro de l'agent.

    Args:
        limit: Nombre de SMS à récupérer (par défaut 10).
    """
    if not is_raphael():
        return _ACCESS_DENIED
    try:
        return await _twilio().call_tool("list_recent_sms", {"limit": limit})
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur lire_sms")
        return f"Je n'ai pas pu lire les SMS : {e}"




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


# ---------------------------------------------------------------------------
# Vérification d'identité
# ---------------------------------------------------------------------------

def _normaliser_mdp(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation, espaces normalises."""
    t = unicodedata.normalize("NFD", texte.strip().lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _reduire_phonetique(mot: str) -> str:
    """Reduction phonetique grossiere du francais, pour comparer des sons.

    Ne vise pas la justesse linguistique : seulement a rapprocher les graphies
    qu'une reconnaissance vocale confond. Consonnes finales muettes retirees,
    doubles simplifiees, graphies equivalentes ramenees a une seule.
    """
    m = mot
    for avant, apres in (("ph", "f"), ("qu", "k"), ("gu", "g"), ("ch", "x"),
                         ("eau", "o"), ("au", "o"), ("ai", "e"), ("ei", "e"),
                         ("ou", "u"), ("y", "i"), ("c", "k"), ("q", "k"),
                         ("z", "s"), ("ss", "s")):
        m = m.replace(avant, apres)
    m = re.sub(r"(.)\1+", r"\1", m)          # doubles -> simple
    m = re.sub(r"[tdsxpgz]+$", "", m)         # consonnes finales muettes
    m = re.sub(r"e$", "", m)                  # e muet final
    return m or mot


def _comparer_mot_de_passe(dit: str, attendu: str) -> str | None:
    """Compare un mot de passe ENTENDU, pas tape. Rend le mode de correspondance.

    Un mot de passe vocal traverse la reconnaissance vocale avant d'arriver ici :
    le 29/08/2026, « fleur de lys » est arrive transcrit « fleur de lit » et la
    comparaison stricte a rejete Raphael avec le bon mot de passe. Comparer au
    caractere pres revient a exiger que le STT soit parfait, ce qu'il n'est pas.

    On tolere donc l'a-peu-pres de transcription, sans ouvrir la porte :
      - meme nombre de mots exige (« fleur de lys » ne matche pas « fleur ») ;
      - au plus UN mot different, et ce mot doit rester tres proche ;
      - similarite globale >= 0.82.
    Sur un secret de trois mots cela laisse passer une syllabe mal entendue,
    pas un mot de passe devine. Le nombre d'essais est borne par ailleurs.
    """
    a, b = _normaliser_mdp(dit), _normaliser_mdp(attendu)
    if not a or not b:
        return None
    if a == b:
        return "exact"
    # Le decoupage en mots est lui-meme une decision du STT : « lune » peut
    # arriver « l une », « aujourd'hui » en « au jour d'hui ». On compare donc
    # aussi les formes collees, reduites phonetiquement — egalite stricte
    # exigee ici, sans quoi « fleurdelys » et « fleurdelune » se ressembleraient
    # trop une fois les espaces retires.
    if _reduire_phonetique(a.replace(" ", "")) == _reduire_phonetique(b.replace(" ", "")):
        return "decoupage des mots approximatif"
    ma, mb = a.split(), b.split()
    # On parle, on ne tape pas : le mot de passe arrive enrobe dans une phrase
    # (« c'est fleur de lys », « alors, fleur de lys, voila »). Exiger le meme
    # nombre de mots rejetait Raphael disant pourtant le bon secret le
    # 29/08/2026. On cherche donc le secret parmi les fenetres de meme longueur
    # de ce qui a ete dit, et on garde la meilleure.
    if len(ma) != len(mb):
        if len(ma) < len(mb) or len(ma) > len(mb) + 12:
            return None  # trop court pour contenir le secret, ou tirade suspecte
        meilleur = None
        for i in range(len(ma) - len(mb) + 1):
            fenetre = " ".join(ma[i:i + len(mb)])
            trouve = _comparer_mot_de_passe(fenetre, b)
            if trouve == "exact":
                return "dit dans une phrase"
            if trouve and meilleur is None:
                meilleur = "dit dans une phrase, transcription approximative"
        return meilleur
    differents = [(x, y) for x, y in zip(ma, mb) if x != y]
    if len(differents) > 1:
        return None
    if differents:
        x, y = differents[0]
        # Comparer les SONS, pas les lettres : c'est une oreille qui a ecrit ce
        # texte. « lys » et « lit » ne partagent qu'une lettre sur trois mais se
        # reduisent tous deux a « li » — c'est exactement la confusion qui a
        # rejete Raphael le 29/08.
        px, py = _reduire_phonetique(x), _reduire_phonetique(y)
        if px != py:
            # Pas le meme son : on tolere encore une lettre de travers, mais
            # l'attaque du mot doit tenir (le STT se trompe sur la fin, rarement
            # sur le debut) — sans quoi « coeur » passerait pour « fleur ».
            if x[:1] != y[:1] or difflib.SequenceMatcher(None, px, py).ratio() < 0.6:
                return None
        return "transcription approximative"
    if difflib.SequenceMatcher(None, a, b).ratio() < 0.82:
        return None
    return "transcription approximative"


@function_tool()
async def verifier_identite(mot_de_passe: str) -> str:
    """Vérifie l'identité de l'appelant via un mot de passe.

    À utiliser uniquement si l'appelant prétend être Raphaël mais appelle d'un
    numéro inconnu.

    Args:
        mot_de_passe: Le mot de passe énoncé par l'appelant.
    """
    expected = os.environ.get("RAPHAEL_VOICE_PASSWORD", "")
    if not expected:
        return "Je ne peux pas vérifier d'identité pour le moment."

    # Anti-force brute : le mot de passe passe par un canal ou l'on peut essayer
    # vite et sans trace. Cinq tentatives par appel, ensuite c'est fini.
    _IDENTITY["essais_mdp"] = _IDENTITY.get("essais_mdp", 0) + 1
    if _IDENTITY["essais_mdp"] > 5:
        logger.warning("Trop de tentatives de mot de passe vocal sur cet appel")
        return ("Trop de tentatives. Je ne peux plus vérifier d'identité pendant "
                "cet appel.")

    verdict = _comparer_mot_de_passe(mot_de_passe, expected)
    if verdict:
        _IDENTITY["is_raphael"] = True
        _IDENTITY["mdp_verifie"] = True
        logger.info("Identite verifiee par mot de passe vocal (%s), palier 2 ouvert", verdict)
        return "Identité confirmée. Bonjour Raphaël."
    return (
        "Mot de passe incorrect. Je ne peux pas vous donner accès aux "
        "informations personnelles."
    )



# ---------------------------------------------------------------------------
# DTMF (tonalites) - navigation de serveurs vocaux
# ---------------------------------------------------------------------------
_DTMF_CODES = {**{str(i): i for i in range(10)}, "*": 10, "#": 11}


@function_tool()
async def envoyer_touches(touches: str) -> str:
    """Envoie une sequence de touches DTMF (tonalites) sur la ligne en cours.

    Utile pour naviguer un serveur vocal automatise : saisir un code postal,
    choisir une option de menu, taper un numero de poste, etc.

    Args:
        touches: La sequence a composer, ex. "30700#" ou "1". Caracteres
            autorises : chiffres 0-9, * et #. Les autres sont ignores.
    """
    try:
        ctx = get_job_context()
        lp = ctx.room.local_participant
        envoyees = []
        for ch in touches:
            code = _DTMF_CODES.get(ch)
            if code is None:
                continue
            await lp.publish_dtmf(code=code, digit=ch)
            envoyees.append(ch)
            await asyncio.sleep(0.25)
        if not envoyees:
            return "Aucune touche valide (chiffres, * ou # uniquement)."
        return "Touches envoyees : " + "".join(envoyees)
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur envoyer_touches")
        return "Touches non envoyees : " + str(e)

async def append_journal_transcript(transcript: str) -> None:
    """Ajoute le transcript a l'entree de journal de CET appel, puis l'ecrit.

    Une seule entree par appel : le titre, les notes prises en cours d'appel et
    le transcript partent ensemble dans le cerveau (append-only).
    """
    texte = transcript or "(aucun echange capte)"
    if not _CALL_STATE.get("journal_title"):
        _CALL_STATE["journal_title"] = "Appel - transcript automatique"
        _CALL_STATE["journal_type"] = "info"
    prev = _CALL_STATE.get("journal_detail") or ""
    _CALL_STATE["journal_detail"] = (prev + "\n\nTranscript de l'appel :\n" + texte)[:20000]
    await flush_journal()


# ---------------------------------------------------------------------------
# Passerelle connecteurs (29/08/2026)
# ---------------------------------------------------------------------------
# Trois outils seulement, quel que soit le nombre de connecteurs : le catalogue
# se decouvre a l'appel au lieu de peser dans le prompt de chaque tour. Le
# palier d'habilitation est recalcule a CHAQUE appel depuis l'etat d'identite
# de la session — jamais mis en cache, pour qu'une identite non confirmee ne
# puisse pas etre rejouee.
from . import connecteurs as _cx


def _palier_courant() -> int:
    """0 = rien, 1 = caller ID de Raphael, 2 = mot de passe verifie.

    Sur un appel SORTANT le caller ID ne vaut rien : c'est le numero que NOUS
    avons compose, il ne dit pas qui a decroche. Le palier 1 y est donc
    inaccessible et seul le mot de passe ouvre quelque chose — ce qui permet
    quand meme a Raphael, quand c'est lui qui decroche, d'atteindre ses donnees
    en se verifiant (constat de son appel de test du 29/08 : « l'appel aux
    outils, ce n'est pas ca encore »).
    """
    if est_sortant():
        return 2 if mdp_verifie() else 0
    if not is_raphael():
        return 0
    return 2 if mdp_verifie() else 1


@function_tool()
async def lister_connecteurs() -> str:
    """Liste les connecteurs de donnees accessibles pendant CET appel.

    A appeler quand tu as besoin d'une information que tes outils habituels ne
    couvrent pas : banque, crypto, fichiers, photos, bibliotheque, messageries,
    fiches du cerveau. Renvoie aussi ce qu'il faudrait pour en ouvrir davantage.
    """
    palier = _palier_courant()
    if palier == 0:
        return ("Aucun connecteur pendant cet appel : ton interlocuteur n'est pas "
                "identifie comme Raphael. C'est voulu, n'insiste pas et ne le "
                "mentionne pas.")
    dispo = _cx.connecteurs_disponibles(palier)
    lignes = [f"- {nom} : {cfg['resume']}" for nom, cfg in sorted(dispo.items())]
    verrouilles = sorted(
        nom for nom, cfg in _cx.REGISTRE.items()
        if cfg["palier"] > palier and _cx._env(nom, "URL")
    )
    txt = "Connecteurs accessibles maintenant :\n" + "\n".join(lignes)
    if verrouilles:
        txt += ("\n\nDemandent le mot de passe vocal (outil verifier_identite) : "
                + ", ".join(verrouilles))
    txt += "\n\nEnsuite : lister_outils(connecteur) puis utiliser_connecteur(...)."
    return txt


@function_tool()
async def lister_outils(connecteur: str) -> str:
    """Detaille les outils d'un connecteur et leurs parametres.

    Args:
        connecteur: Nom rendu par lister_connecteurs (ex. "qonto", "memoire").
    """
    palier = _palier_courant()
    dispo = _cx.connecteurs_disponibles(palier)
    if connecteur not in dispo:
        return (f"« {connecteur} » n'est pas accessible a ce niveau. "
                "Appelle lister_connecteurs pour voir ce qui est ouvert.")
    try:
        distants = await _cx.lister_outils_distants(connecteur)
    except Exception as e:  # noqa: BLE001
        logger.exception("lister_outils %s", connecteur)
        return f"Connecteur {connecteur} injoignable : {e}"
    lignes = []
    for o in distants:
        nom = o.get("name", "")
        ok, _ = _cx.outil_autorise(connecteur, nom)
        if not ok:
            continue  # ecriture ou outil non reconnu comme lecture : invisible
        params = ", ".join((o.get("inputSchema") or {}).get("properties", {}).keys())
        lignes.append(f"- {nom}({params}) : {(o.get('description') or '')[:150]}")
    if not lignes:
        return f"Aucun outil autorise expose par {connecteur}."
    return f"Outils de {connecteur} :\n" + "\n".join(lignes)


@function_tool()
async def utiliser_connecteur(connecteur: str, outil: str, arguments: str = "{}") -> str:
    """Appelle un outil d'un connecteur et renvoie son resultat.

    Previens ton interlocuteur que tu regardes ("je vous regarde ca") AVANT
    d'appeler : la reponse peut prendre quelques secondes.

    Args:
        connecteur: Nom du connecteur (voir lister_connecteurs).
        outil: Nom exact de l'outil (voir lister_outils).
        arguments: Arguments au format JSON, ex. '{"limit": 5}'. "{}" si aucun.
    """
    palier = _palier_courant()
    dispo = _cx.connecteurs_disponibles(palier)
    if connecteur not in dispo:
        if connecteur in _cx.REGISTRE and palier == 1:
            return (f"« {connecteur} » touche a des donnees sensibles : demande a "
                    "Raphael son mot de passe vocal, puis appelle verifier_identite.")
        return (f"« {connecteur} » n'est pas accessible pendant cet appel. "
                "Appelle lister_connecteurs.")
    ok, detail = _cx.outil_autorise(connecteur, outil)
    if not ok:
        logger.warning("Passerelle : refus %s/%s (%s)", connecteur, outil, detail)
        return f"Refuse : {detail}"
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        if not isinstance(args, dict):
            return "Les arguments doivent etre un objet JSON, ex. '{\"limit\": 5}'."
    except Exception:  # noqa: BLE001
        return f"Arguments JSON illisibles : {arguments[:120]}"
    logger.info("Passerelle : %s/%s palier=%d args=%s", connecteur, outil, palier,
                str(args)[:120])
    try:
        res = await _cx.appeler_outil(connecteur, outil, args)
    except Exception as e:  # noqa: BLE001
        logger.exception("Passerelle %s/%s", connecteur, outil)
        return f"Le connecteur {connecteur} a echoue : {e}"
    # Au telephone, un pave est inexploitable : on borne et on le dit.
    if len(res) > 3000:
        res = res[:3000] + "\n[...] (resultat tronque, affine ta demande)"
    return res or "(reponse vide)"
