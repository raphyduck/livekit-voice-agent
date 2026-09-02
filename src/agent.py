import asyncio
import json
import re
import logging
import time
import os

import httpx

from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import (
    Agent, AgentSession, EndpointingOptions, RoomInputOptions, RoomOutputOptions,
    TurnHandlingOptions, inference, mcp, metrics,
)
from livekit.agents.voice.turn import InterruptionOptions, PreemptiveGenerationOptions
from livekit.plugins import (
    anthropic, cartesia, deepgram, noise_cancellation, openai, silero,
)
from livekit.agents.voice.amd import AMD

from .system_prompt import get_system_prompt
from .tools import (
    add_task,
    create_calendar_event,
    end_call,
    get_calendar_events,
    get_current_datetime,
    get_today_tasks,
    get_unread_emails,
    is_raphael,
    read_brain,
    reset_identity,
    set_direction_sortante,
    send_email,
    send_sms,
    lire_sms,
    verifier_identite,
    write_journal,
    envoyer_touches,
    journal_was_written,
    reset_call_state,
    write_journal_raw,
    journal_page_id,
    append_journal_transcript,
    lister_connecteurs,
    lister_outils,
    utiliser_connecteur,
)

load_dotenv()
logger = logging.getLogger("voice-agent")

# --- Correctif Haiku 4.5 (et autres Claude 4.x récents) -------------------
# Le plugin livekit-plugins-anthropic 1.6.0 ne connaît pas claude-haiku-4-5.
# Sa liste _NO_PREFILL_PATTERNS (modèles qui NE supportent PAS le prefill /
# message assistant final) ne contient que sonnet-4-6 et opus-4-6. Pour tout
# autre modèle, le plugin laisse un message assistant en position finale
# (prefill), ce qui casse silencieusement la génération avec Haiku 4.5
# (l'agent entend mais ne répond jamais). On élargit la liste pour couvrir
# les Claude 4.x récents qui se comportent comme 4.6 côté prefill.
try:
    import livekit.plugins.anthropic.llm as _anthropic_llm
    _extra_no_prefill = (
        "claude-haiku-4-5",
        "claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-mythos",
    )
    _existing = tuple(_anthropic_llm._NO_PREFILL_PATTERNS)
    _merged = _existing + tuple(p for p in _extra_no_prefill if p not in _existing)
    _anthropic_llm._NO_PREFILL_PATTERNS = _merged
    logger.info("Patch prefill appliqué, modèles no-prefill: %s", _merged)
except Exception:  # noqa: BLE001
    logger.exception("Échec du patch _NO_PREFILL_PATTERNS")

# --- Claude 5 : contraintes API (constat 29/08/2026) -----------------------
# - `temperature` non-defaut => 400 "deprecated for this model".
# - thinking adaptatif ACTIF par defaut => latence imprevisible au telephone,
#   on le desactive explicitement a chaque requete via extra_kwargs.
# - prefill (message assistant final) refuse => couvert par le patch ci-dessus.
_CLAUDE5_PREFIXES = ("claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-mythos")


class _Claude5LLM(anthropic.LLM):
    def chat(self, **kwargs):
        extra = dict(kwargs.pop("extra_kwargs", None) or {})
        extra.setdefault("thinking", {"type": "disabled"})
        return super().chat(extra_kwargs=extra, **kwargs)


def _make_llm(model: str):
    """Construit le LLM de l'appel, chez Anthropic ou via la passerelle LiteLLM.

    Tout modele qui n'est pas un « claude-* » part sur LiteLLM (127.0.0.1:4000),
    la meme passerelle que LibreChat : cela ouvre les 18 modeles du serveur sans
    ecrire un client par fournisseur.
    """
    if "/" in model:
        # Modeles heberges LiveKit Inference (ex. google/gemma-4-31b-it) :
        # gateway OpenAI-compatible de LiveKit Cloud, jeton minté automatiquement
        # depuis LIVEKIT_API_KEY/SECRET — aucun compte fournisseur ni passerelle.
        return inference.LLM(model=model, extra_kwargs={"temperature": 0.6})
    if not model.startswith("claude"):
        base = os.environ.get("LITELLM_URL", "http://127.0.0.1:4000/v1")
        cle = os.environ.get("LITELLM_KEY", "")
        if not cle:
            logger.error("LITELLM_KEY absente, repli sur claude-sonnet-5")
            return _Claude5LLM(model="claude-sonnet-5", caching="ephemeral")
        return openai.LLM(model=model, base_url=base, api_key=cle, temperature=0.6)
    if model.startswith(_CLAUDE5_PREFIXES):
        return _Claude5LLM(model=model, caching="ephemeral")
    return anthropic.LLM(model=model, temperature=0.7, caching="ephemeral")


MODELES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "terra": "gpt-5.6-terra",
    "gemma": "google/gemma-4-31b-it",
    "luna": "gpt-5.6-luna",
}
# Defaut passe a gemma (LiveKit Inference) le 02/09/2026 sur decision de
# Raphael, apres rejeu du banc : TTFT 0,33-0,38 s (vs 0,70 haiku), outils 6/6
# sur deux passes (haiku 5/6, terra 4/6 le meme jour), completions outils en
# 0,3-0,5 s. Points a surveiller en appel reel : restitution des numeros
# dictes, glissements tu/vous. Retour arriere : LLM_MODEL=claude-haiku-4-5.
# Precedent defaut, valide par Raphael en appel reel le 29/08/2026 :
# Mesure decisive : le TTFT doit se comparer AVEC les outils, puisque l'agent
# en a toujours. haiku-4-5 ne paie aucun surcout d'outillage (0,64 s avec
# 9 outils, contre 1,29 s pour terra qui en paie 0,69). En appel reel :
# TTFT median 1,09 s sur 16 tours.
MODELE_DEFAUT = "gemma"


def _resoudre_modele(demande: str | None = None) -> str:
    """Nom d'API du modele pour cet appel. `demande` vient du metadata (sortant)."""
    m = demande or os.environ.get("LLM_MODEL") or os.environ.get("VOICE_LLM", MODELE_DEFAUT)
    return MODELES.get(
        m, m if (m.startswith("claude") or m.startswith("gpt") or "/" in m) else MODELES[MODELE_DEFAUT]
    )


# Délai de silence (secondes) avant relance puis raccrochage.
INACTIVITY_TIMEOUT = 10.0
# Quand l'agent vient de dire « un instant, je regarde », le silence qui suit
# est celui de quelqu'un qui attend une reponse promise : on lui laisse le
# temps avant de demander s'il est toujours la.
INACTIVITY_PROMESSE = 30.0
# Formules par lesquelles l'agent annonce qu'il cherche quelque chose.
_MOTIFS_ATTENTE = re.compile(
    r"\b(un instant|une seconde|deux secondes|je regarde|je vérifie|je verifie|"
    r"je consulte|je cherche|je vais voir|laisse-moi voir|je te vérifie|"
    r"je vous vérifie|je m'en occupe|patiente|patientez)\b",
    re.IGNORECASE,
)

# Jeu COMPLET : appel entrant de Raphael identifie. Il peut tout demander.
TOOLS = [
    get_calendar_events,
    create_calendar_event,
    get_unread_emails,
    send_email,
    get_today_tasks,
    add_task,
    read_brain,
    write_journal,
    send_sms,
    lire_sms,
    get_current_datetime,
    end_call,
    verifier_identite,
    envoyer_touches,
    # Passerelle : trois outils, quel que soit le nombre de connecteurs.
    lister_connecteurs,
    lister_outils,
    utiliser_connecteur,
]

# Jeu RESTREINT : appel sortant, ou entrant d'un inconnu. Tout ce qui est retire
# ici serait de toute facon refuse (la passerelle rend palier 0, les outils
# sensibles renvoient _ACCESS_DENIED) : on paie leur schema a chaque tour pour
# rien. Mesure du 29/08 : ~5000 des 7000 tokens de prompt venaient des schemas.
# verifier_identite reste, sans quoi un appelant legitime n'aurait aucun moyen
# de se faire reconnaitre et de debloquer le jeu complet.
TOOLS_RESTREINT = [
    get_current_datetime,
    read_brain,
    write_journal,
    end_call,
    verifier_identite,
    envoyer_touches,
    # Passerelle : elle ne donne RIEN tant que le mot de passe n'est pas
    # verifie (palier 0 sur un appel sortant), mais sans ces trois outils
    # Raphael ne peut rien consulter quand c'est l'agent qui l'appelle.
    lister_connecteurs,
    lister_outils,
    utiliser_connecteur,
]


async def _probe_mcp(url: str, headers: dict | None = None, timeout: float = 6.0) -> bool:
    """Verifie qu'un serveur MCP tient un vrai handshake avant de l'attacher.

    Un serveur MCP casse l'agent des l'initialisation de la session : le setup
    du toolset leve, l'appel demarre a moitie et l'interlocuteur tombe sur un
    agent inutilisable. Le filet doit donc etre pose AVANT l'attachement.

    L'ancienne sonde faisait un GET et acceptait tout statut < 400. Insuffisant,
    constate le 29/08/2026 : le serveur WhatsApp repond 200 au GET mais rend un
    Content-Type VIDE sur le POST initialize, et le client MCP refuse
    (« Expected response header Content-Type to contain 'text/event-stream' »).
    L'agent partait donc en erreur sur chaque appel entrant. On rejoue ici le
    vrai premier echange : POST initialize, et le type de contenu doit etre
    exploitable.
    """
    probe_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if headers:
        probe_headers.update(headers)
    corps = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "voice-agent-probe", "version": "1.0"},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=probe_headers, json=corps)
        if resp.status_code >= 400:
            logger.warning("MCP %s repond %s, ignore", url, resp.status_code)
            return False
        type_contenu = (resp.headers.get("content-type") or "").lower()
        if not ("json" in type_contenu or "event-stream" in type_contenu):
            logger.warning(
                "MCP %s : Content-Type inexploitable (%r), ignore pour ne pas "
                "casser la session", url, type_contenu)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP %s injoignable (%s), ignore", url, exc)
        return False


async def _build_mcp_servers() -> list:
    """Serveurs MCP self-hosted attachés UNIQUEMENT pour Raphaël identifié.

    WhatsApp uniquement. IMAP reste géré par les outils codés
    (get_unread_emails/send_email) pour éviter le doublon. Bitwarden est exclu.

    Le MCP navigateur (human-browser) a été retiré le 30/08/2026 : ses 23 outils
    portaient à 40 le nombre d'outils exposés en appel entrant, ce qui faisait
    dépasser le budget de complexité des schémas chez Anthropic (400 « Schema is
    too complex », agent muet). Piloter un navigateur à la voix n'a par ailleurs
    guère de sens.

    Chaque serveur est probé avant d'être attaché : un serveur injoignable est
    ignoré (avec log) au lieu de faire planter le setup des outils.
    """
    servers = []
    wa_url = os.environ.get("WA_MCP_URL")
    if wa_url:
        # WhatsApp : le secret est déjà dans l'URL, pas de header.
        if await _probe_mcp(wa_url):
            servers.append(
                mcp.MCPServerHTTP(
                    url=wa_url,
                    client_session_timeout_seconds=10,
                )
            )
    return servers


async def entrypoint(ctx: agents.JobContext):
    logger.info("Agent démarré pour la room: %s", ctx.room.name)
    await ctx.connect()

    # --- Enregistrement audio (egress -> MinIO S3) en mode capture OTP --------
    # LiveKit Cloud ecrit le fichier dans le bucket. On l'utilise pour transcrire
    # les OTP a posteriori (Whisper), fiable la ou la transcription live echoue.
    if os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True"):
        try:
            s3 = api.S3Upload(
                access_key=os.environ["S3_ACCESS_KEY"],
                secret=os.environ["S3_SECRET_KEY"],
                bucket=os.environ["S3_BUCKET"],
                endpoint=os.environ["S3_ENDPOINT"],
                region=os.environ.get("S3_REGION", "us-east-1"),
                force_path_style=(os.environ.get("S3_FORCE_PATH_STYLE","false").strip() in ("1","true","True")),
            )
            req = api.RoomCompositeEgressRequest(
                room_name=ctx.room.name,
                audio_only=True,
                file_outputs=[api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=f"otp/{ctx.room.name}.ogg",
                    s3=s3,
                )],
            )
            lkapi = api.LiveKitAPI()
            info = await lkapi.egress.start_room_composite_egress(req)
            logger.info("Egress OTP demarre: egress_id=%s -> s3://%s/otp/%s.ogg",
                        getattr(info, "egress_id", "?"), os.environ["S3_BUCKET"], ctx.room.name)
        except Exception:
            logger.exception("Echec demarrage egress OTP (on continue sans enregistrement)")

    # --- Identification de l'appelant (caller ID, authentification faible) ----
    # NB : le caller ID SIP est spoofable et un mot de passe parlé est faible ;
    # c'est un compromis assumé pour un usage perso, pas une sécurité forte.
    reset_identity(False)  # repartir non identifié à chaque appel
    reset_call_state()
    raphael_phone = os.environ.get("RAPHAEL_PHONE", "")
    caller = ""
    participant = None
    try:
        participant = await ctx.wait_for_participant()
        caller = participant.attributes.get("sip.phoneNumber", "")
    except Exception:  # noqa: BLE001
        logger.exception("Impossible de lire le participant appelant")
    logger.info("Appel entrant de: %s", caller or "inconnu")
    reset_identity(bool(raphael_phone) and caller == raphael_phone)

    # --- Contexte d'appel sortant (metadata JSON portée par la room) ----------
    # voicecallmcp lance un appel sortant en attachant un metadata
    # { direction, scenario, objectif, contexte }. On l'utilise pour adapter le
    # prompt et l'ouverture de l'agent. En entrant, ce metadata est absent.
    call_ctx = {}
    try:
        raw = ctx.room.metadata or ""
        if raw:
            call_ctx = json.loads(raw)
    except Exception:  # noqa: BLE001
        logger.exception("metadata room illisible")

    is_outbound = call_ctx.get("direction") == "outbound"
    set_direction_sortante(is_outbound)
    objectif = call_ctx.get("objectif", "")
    scenario = call_ctx.get("scenario", "")
    contexte_appel = call_ctx.get("contexte", "")
    consignes_perso = call_ctx.get("instructions", "")

    # Modele par appel : entrant => Sonnet (qualite, besoin imprevisible) ;
    # sortant => choix passe dans le metadata (defaut Haiku, sinon LLM_MODEL).
    llm_model = _resoudre_modele(call_ctx.get("model") if is_outbound else None)
    logger.info("LLM pour cet appel: %s (outbound=%s)", llm_model, is_outbound)

    # --- Compte-rendu garanti : callback shutdown enregistre AU PLUS TOT ------
    # Fix 2026-07-20 : sur un rejet immediat (USER_REJECTED <1s), la session se
    # fermait avant d'atteindre l'ancien add_shutdown_callback -> aucun journal.
    # On enregistre ici un wrapper ; _journal_fin_appel s'y branchera une fois
    # session/live_transcript disponibles.
    _fin_appel_impl = {"fn": None}

    async def _fin_appel_early():
        fn = _fin_appel_impl["fn"]
        if fn is not None:
            await fn()
            return
        if not is_outbound:
            return
        try:
            await write_journal_raw(
                "Appel sortant termine avant initialisation complete",
                ("Objectif : " + (objectif or "?")[:300]
                 + " — aucun echange (rejet ou raccroche immediat probable, "
                   "voir logs livekit-voice-agent)."),
                "erreur",
            )
        except Exception:
            logger.exception("Echec journal early-shutdown")

    ctx.add_shutdown_callback(_fin_appel_early)

    prompt = get_system_prompt()
    if is_outbound:
        consignes = {
            "rdv": "Tu appelles pour prendre, décaler ou annuler un rendez-vous.",
            "rappel": "Tu appelles pour transmettre ou obtenir une information.",
            "message": "Tu appelles pour laisser un message à transmettre.",
            "relance": "Tu appelles pour relancer sur un sujet en attente.",
        }.get(scenario, "")
        prompt = prompt + f"""

CONTEXTE DE CET APPEL (SORTANT) :
- C'est TOI qui appelles. Ton interlocuteur est le plus souvent un professionnel occupé, donc sois efficace, mais reste une vraie personne au bout du fil : naturelle, chaleureuse, avenante.
- OUVERTURE : présente-toi brièvement et donne l'objet de l'appel dès tes premières phrases, sans tourner autour du pot. Exemple : « Bonjour, Claude à l'appareil, j'appelle de la part de Raphaël Nicolle. Il a une fuite sous l'évier de sa salle de bain et cherche quelqu'un pour intervenir cette semaine, est-ce que ce serait possible ? » Tu peux dire bonjour et demander si tu tombes bien avant d'enchaîner, ça reste courtois.
- N'annonce PAS spontanément que tu es une IA, et ne t'excuse PAS par avance d'appeler. Pas de justification non sollicitée, mais pas de sécheresse non plus.
- Pendant l'échange : sois chaleureuse et fluide, comme une bonne assistante au téléphone. De petites marques de politesse et de connivence sont les bienvenues (« très bien », « parfait, je note », « avec plaisir »). Efficace ne veut pas dire expéditive ni robotique.
- Si l'interlocuteur demande pourquoi il a affaire à une IA ou s'agace : réponds simplement en une phrase, avec le sourire dans la voix, sans t'excuser longuement, et propose aussitôt une issue concrète. Exemple : « Tout à fait, je suis l'assistante de Raphaël et je l'appelle pour lui faire gagner du temps ; si vous préférez, il peut vous rappeler lui-même sans souci. »
- {consignes}
- Objectif précis : {objectif}
- Informations utiles : {contexte_appel or "aucune"}
- Mène la conversation vers cet objectif avec efficacité ET chaleur, mais SOIS BRÈVE : une à deux phrases par tour grand maximum. La chaleur passe par le ton et de petits mots simples (« parfait », « avec plaisir »), jamais par des phrases plus longues. Évite les tournures d'empathie plaquées et corporate (« je comprends tout à fait votre réaction »)."""
        if consignes_perso:
            prompt += f"- Stratégie spécifique pour cet appel : {consignes_perso}\n"

    # --- Serveurs MCP self-hosted (gating sécurité) --------------------------
    # Attachés UNIQUEMENT si l'appelant est identifié comme Raphaël. Appelant
    # inconnu OU appel sortant (l'appelé n'est pas Raphaël) => aucun serveur MCP,
    # pour ne jamais exposer ces outils à un tiers.
    mcp_servers = (await _build_mcp_servers()) if (is_raphael() and not is_outbound) else []
    logger.info("Serveurs MCP attachés: %d (is_raphael=%s, sortant=%s, outils=%d)",
                len(mcp_servers), is_raphael(), is_outbound,
                len(TOOLS if (is_raphael() and not is_outbound) else TOOLS_RESTREINT))

    session = AgentSession(
        stt=(
            deepgram.STT(
                model="nova-3",
                base_url="https://api.eu.deepgram.com/v1/listen",
                # Les services OTP (WhatsApp) parlent anglais : "press 7".
                # Surchargeable via OTP_STT_LANGUAGE si besoin.
                language=os.environ.get("OTP_STT_LANGUAGE", "multi"),
                # Profil capture OTP : chiffres explicites, pas de mise en forme
                # qui segmente en phrases, pour ne pas perdre le 1er chiffre ni couper.
                smart_format=False,
                punctuate=False,
                numerals=True,
            )
            if os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
            else deepgram.STTv2(
                # Flux : STT conversationnel avec detection de fin de tour
                # NATIVE (signaux appris, pas des seuils de silence). Le
                # multilingue couvre le francais. Endpoint EU (/v2/listen).
                model="flux-general-multi",
                language_hint=["fr"],
                base_url="wss://api.eu.deepgram.com/v2/listen",
                # eager => generation preemptive des qu'un EoT probable ;
                # eot releve en consequence (recommandation du plugin).
                # Mesure du 29/08 : end_of_utterance_delay de 0,4 a 1,25 s
                # avec 0.5/0.8. On avance les deux seuils d'un cran : la
                # generation preemptive part plus tot et la fin de tour est
                # actee plus vite, sans tomber sous les valeurs ou Flux
                # commence a couper la parole.
                eager_eot_threshold=0.4,
                eot_threshold=0.7,
                # Filet de securite quand Flux n'est sur de rien : il conclut la
                # fin de tour sur silence pur. Defaut du plugin 3000 ms, soit
                # trois secondes de blanc au telephone. 1200 ms suffit : au-dela
                # d'une seconde de silence, l'interlocuteur a fini.
                eot_timeout_ms=int(os.environ.get("FLUX_EOT_TIMEOUT_MS", "1200")),
            )
        ),
        # Modèle décidé par appel : entrant => Sonnet ; sortant => metadata.
        # Claude 5 : pas de temperature, thinking désactivé (voir _make_llm).
        llm=_make_llm(llm_model),
        tts=cartesia.TTS(
            model="sonic-3.6",
            voice=os.environ["CARTESIA_VOICE_ID"],
            language="fr",
        ),
        vad=silero.VAD.load(),
        # --- Gestion des tours de parole (API 1.6 : turn_handling unifié) -----
        # Remplace les anciens kwargs dépréciés (turn_detection / min|max_endpointing
        # _delay / preemptive_generation), qui se battaient entre eux.
        turn_handling=TurnHandlingOptions(
            # Mode normal : fin de tour decidee par Flux ("stt"), qui la
            # detecte nativement. Mode OTP : nova-3 n'a pas d'EoT natif, on
            # garde le detecteur audio v1 (Cloud), defauts serveur.
            turn_detection=(
                inference.TurnDetector(version="v1")
                if os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
                else "stt"
            ),
            # --- Endpointing : le poste de latence n°1, corrige le 29/08/2026 ---
            # Diagnostic. Meme en mode "stt", l'agent attend `endpointing.min_delay`
            # APRES que Flux a decide la fin de tour (audio_recognition.py :
            # `endpointing_delay = self._endpointing.min_delay`). En mode dynamique
            # ce min_delay n'est pas la valeur passee ici : il est APPRIS sur les
            # pauses de l'interlocuteur par un filtre exponentiel (alpha 0.9) qui
            # monte vers max_delay. Mesure sur deux appels reels du 29/08, meme
            # configuration :
            #     appel 11h24 — EOU median 0,72 s dont 0,46 s de transcription
            #     appel 11h57 — EOU median 1,13 s dont 0,38 s de transcription
            # La transcription est stable ; c'est l'attente apprise qui passe de
            # 0,26 s a 0,75 s parce que Raphael avait hesite (phrases hachees),
            # et le filtre redescend lentement. D'ou l'impression de lenteur
            # variable d'un appel a l'autre.
            #
            # Correction. Cette attente apprise fait doublon avec Flux, qui a deja
            # tranche semantiquement (eot_threshold 0.7) : on garde un delai FIXE
            # et court, juste de quoi laisser arriver une transcription tardive.
            # Le mode dynamique reste justifie sans detecteur semantique.
            endpointing=(
                # Mode OTP : endpointing long et non-dynamique pour ne PAS clore
                # l'énoncé entre les chiffres dictés (sinon la 2e moitié est perdue).
                EndpointingOptions(mode="fixed", min_delay=2.5, max_delay=6.0)
                if os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
                else EndpointingOptions(
                    mode="fixed",
                    min_delay=float(os.environ.get("EOU_MIN_DELAY", "0.15")),
                    max_delay=float(os.environ.get("EOU_MAX_DELAY", "0.6")),
                )
            ),
            # Interruptions ADAPTATIVES (barge-in propre) : c'était désactivé par
            # défaut en prod, d'où l'impression de blancs/coupures. On l'active.
            interruption=InterruptionOptions(enabled=True, mode="adaptive"),
            # Génération préemptive : le LLM démarre pendant que l'utilisateur finit.
            preemptive_generation=PreemptiveGenerationOptions(enabled=True),
        ),
        # Serveurs MCP (WhatsApp seul depuis le 30/08), uniquement pour Raphaël identifié.
        mcp_servers=mcp_servers,
    )

    # --- Capture live du transcript (fix bug 02/07/2026) ------------------
    # session.history.items lu au shutdown peut manquer le dernier tour si
    # l'appel se coupe brutalement (raccroche interlocuteur) avant que le
    # tour soit "committed". conversation_item_added est émis de façon
    # synchrone dès qu'un item (user OU assistant) est ajouté au contexte :
    # on l'utilise comme source de vérité principale pour le transcript.
    live_transcript: list[str] = []

    def _set_attente_promise(valeur: bool) -> None:
        nonlocal attente_promise
        attente_promise = valeur

    @session.on("conversation_item_added")
    def _on_conv_item(ev):
        try:
            item = ev.item
            role = getattr(item, "role", "")
            if role not in ("user", "assistant"):
                return
            txt = getattr(item, "text_content", None)
            if not txt:
                return
            qui = "Agent" if role == "assistant" else "Interlocuteur"
            live_transcript.append(f"{qui}: {txt}")
            nonlocal_attente = _MOTIFS_ATTENTE.search(txt or "") is not None
            if role == "assistant":
                _set_attente_promise(nonlocal_attente)
            elif role == "user":
                _set_attente_promise(False)
        except Exception:
            logger.exception("Erreur capture live transcript (conversation_item_added)")

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=prompt,
            tools=(TOOLS if (is_raphael() and not is_outbound) else TOOLS_RESTREINT),
        ),
        # BVCTelephony : annulation de bruit + voix de fond, optimisee pour le
        # telephone (plugin livekit-plugins-noise-cancellation, LiveKit Cloud).
        # delete_room_on_close ferme la room (et coupe la ligne) à la fin.
        room_input_options=RoomInputOptions(
            delete_room_on_close=True,
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
        # audio_enabled=True est le fix central : force la publication du track
        # audio de sortie de l'agent dans la room.
        room_output_options=RoomOutputOptions(
            audio_enabled=True,
            transcription_enabled=True,
        ),
    )

    # --- Relance puis raccrochage sur silence prolongé -------------------
    # On arme un minuteur quand l'agent repasse en écoute ; on l'annule dès que
    # l'utilisateur reparle. Après INACTIVITY_TIMEOUT de silence on relance une
    # fois, puis au silence suivant on raccroche poliment.
    inactivity_task: asyncio.Task | None = None
    relance_count = 0
    # Outils en cours d'execution. Le minuteur de silence ne doit pas tourner
    # pendant qu'un outil travaille : c'est l'agent qui fait patienter, pas
    # l'interlocuteur qui s'absente (constat de Raphael, 29/08/2026 : l'agent
    # disait « un instant, je regarde » puis, quelques secondes plus tard,
    # « vous etes toujours la ? »).
    outils_en_cours = 0
    etat_agent = "initializing"
    # L'agent vient-il de promettre un resultat (« un instant, je regarde ») ?
    # Filet pour le cas ou le modele annonce l'attente SANS lancer l'outil dans
    # le meme tour : mesure du 29/08, 20 s se sont ecoulees entre l'annonce et
    # la plainte de Raphael, sans le moindre appel d'outil. La vraie correction
    # est dans le prompt ; ici on evite au moins de le relancer alors qu'il
    # croit legitimement qu'on travaille pour lui.
    attente_promise = False
    # Le raccrochage est un point de non-retour : une fois engage, plus aucune
    # relance, plus aucun minuteur. Sans ce verrou, l'agent a repete « je vais
    # raccrocher » TREIZE fois en trois minutes sur un repondeur, le 29/08/2026.
    raccrochage_engage = False

    # --- Mesure de latence : loguer les métriques de chaque tour ----------
    @session.on("metrics_collected")
    def _on_metrics(ev):
        try:
            metrics.log_metrics(ev.metrics)
        except Exception:  # noqa: BLE001
            logger.debug("metrics: %s", getattr(ev, "metrics", None))

    async def _hangup() -> None:
        """Coupe la ligne. Appelable plusieurs fois, ne raccroche qu'une.

        Deux pieges, tous deux constates le 29/08/2026 sur un appel tombe sur
        un repondeur :
        - `wait_for_idle()` sans borne ne rend JAMAIS la main si l'agent se
          remet a parler entre-temps ; le raccrochage restait suspendu pendant
          que le minuteur relancait, et l'agent a repete « je vais raccrocher »
          treize fois en trois minutes, jusqu'a saturer la messagerie.
        - supprimer la room peut echouer ; il faut alors une seconde voie de
          sortie, sinon l'appel ne se termine pas du tout.
        """
        nonlocal raccrochage_engage
        if raccrochage_engage:
            return
        raccrochage_engage = True
        _cancel_inactivity()
        try:
            # Laisser finir la phrase en cours, mais jamais plus de 6 secondes.
            try:
                await asyncio.wait_for(session.wait_for_idle(), timeout=6.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
            await asyncio.sleep(0.3)  # marge réseau dernier paquet audio
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:  # noqa: BLE001
            logger.exception("Échec de la suppression de room, arrêt du job à la place")
            try:
                ctx.shutdown(reason="raccrochage")
            except Exception:  # noqa: BLE001
                logger.exception("Échec de l'arrêt du job")

    async def _inactivity_watch() -> None:
        nonlocal relance_count
        try:
            await asyncio.sleep(
                INACTIVITY_PROMESSE if attente_promise else INACTIVITY_TIMEOUT
            )
        except asyncio.CancelledError:
            return
        # Un outil a pu demarrer pendant l'attente : dernier controle avant de
        # parler, sinon on relance en plein milieu d'une recherche.
        if _agent_occupe():
            _arm_inactivity()
            return
        if raccrochage_engage:
            return
        relance_count += 1
        if relance_count == 1:
            await session.say("Vous êtes toujours là ?", allow_interruptions=True)
            # Réarmer pour laisser une 2e chance APRÈS la relance.
            _arm_inactivity()
        else:
            # Le minuteur est coupe AVANT de parler ; pendant la phrase d'adieu
            # l'agent est en « speaking », donc _agent_occupe() empeche tout
            # rearmement, et _hangup pose ensuite le verrou definitif.
            _cancel_inactivity()
            try:
                await session.say(
                    "Je vais raccrocher, n'hésitez pas à me rappeler. Au revoir.",
                    allow_interruptions=False,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Adieu non prononcé, on raccroche quand même")
            await _hangup()

    def _agent_occupe() -> bool:
        """L'agent travaille-t-il ? Alors le silence d'en face est normal.

        Deux raisons de ne pas compter : un outil tourne, ou l'agent parle /
        reflechit. Le cas de l'outil ne se deduit PAS de l'etat : livekit met
        l'agent en « listening » pendant un appel d'outil quand il n'a pas de
        parole de fond (agent_activity.py : `"thinking" if
        self._background_speeches else "listening"`), ce qui armait le minuteur
        au milieu d'une recherche. D'ou le comptage explicite des outils.
        """
        return (
            raccrochage_engage
            or outils_en_cours > 0
            or etat_agent in ("speaking", "thinking", "initializing")
        )

    def _arm_inactivity() -> None:
        nonlocal inactivity_task
        if _agent_occupe():
            # On reviendra armer quand l'agent aura fini (agent_state_changed
            # « listening » ou fin du dernier outil).
            return
        current = asyncio.current_task()
        if inactivity_task and inactivity_task is not current and not inactivity_task.done():
            inactivity_task.cancel()
        inactivity_task = asyncio.create_task(_inactivity_watch())

    def _cancel_inactivity() -> None:
        nonlocal inactivity_task, relance_count
        relance_count = 0
        if inactivity_task and not inactivity_task.done():
            inactivity_task.cancel()
        inactivity_task = None

    # Logique basée UNIQUEMENT sur l'état de l'UTILISATEUR :
    # - dès qu'il parle (speaking) -> on annule tout (silence rompu, compteur reset)
    # - quand il arrête (listening) ou s'absente (away) -> on (ré)arme le minuteur
    # On NE se base PAS sur l'état de l'agent, sinon le minuteur démarre dès que
    # l'agent finit de parler, avant même que l'utilisateur ait eu le temps de répondre.
    @session.on("user_state_changed")
    def _on_user_state(ev):
        new = getattr(ev, "new_state", None)
        if new == "speaking":
            _cancel_inactivity()
        elif new in ("listening", "away"):
            # L'utilisateur s'est tu : (re)lancer le décompte de silence.
            _arm_inactivity()

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        nonlocal etat_agent
        new = getattr(ev, "new_state", None)
        etat_agent = new or etat_agent
        if new == "speaking":
            # L'agent parle : ce n'est pas un silence, on suspend le décompte.
            if inactivity_task and not inactivity_task.done():
                inactivity_task.cancel()
        elif new == "listening":
            # L'agent a fini de parler et attend l'utilisateur : (re)lancer le décompte.
            # (couvre le cas où l'utilisateur était déjà silencieux, donc aucun
            #  user_state_changed ne se déclenche après la réponse de l'agent)
            _arm_inactivity()

    # --- Outils en cours : suspendre le minuteur de silence pendant le travail -
    @session.on("tool_execution_updated")
    def _on_tool_execution(ev):
        nonlocal outils_en_cours
        try:
            type_maj = getattr(getattr(ev, "update", None), "type", "")
            if type_maj == "tool_call_started":
                outils_en_cours += 1
                _cancel_inactivity()
            elif type_maj == "tool_call_ended":
                outils_en_cours = max(0, outils_en_cours - 1)
                if outils_en_cours == 0:
                    # L'outil a rendu : l'agent va parler, le minuteur repartira
                    # quand il repassera en ecoute.
                    _arm_inactivity()
        except Exception:
            logger.exception("suivi des outils en cours")

    # --- Mode OTP : suivre l'instruction vocale et presser la touche demandee ---
    # Certains services (WhatsApp) demandent "press N" / "appuyez sur N" avant de
    # dicter le code. On detecte le chiffre dans la transcription et on envoie le
    # DTMF correspondant. _otp_pressed evite d'appuyer plusieurs fois pour rien.
    _otp_pressed: set[str] = set()
    # Deepgram livre souvent "press" et le chiffre dans DEUX segments distincts.
    # On garde un tampon glissant des derniers segments et on cherche dedans.
    _otp_buf: list[str] = []

    _WORD2DIGIT = {
        "zero": "0", "un": "1", "one": "1", "deux": "2", "two": "2",
        "trois": "3", "three": "3", "quatre": "4", "four": "4",
        "cinq": "5", "five": "5", "six": "6", "sept": "7", "seven": "7",
        "huit": "8", "eight": "8", "neuf": "9", "nine": "9",
    }

    def _extract_press_digit(t: str):
        low = t.lower()
        # motifs: "press 7", "appuyez sur 7", "tapez 7", "press seven", "appuyez sur sept"
        m = re.search(r"(?:press|appuye[sz]?\s+sur|appuyer\s+sur|tape[sz]?|composez)\s+(?:the\s+)?(?:touche\s+|key\s+|number\s+|chiffre\s+|le\s+)?([0-9]|zero|un|one|deux|two|trois|three|quatre|four|cinq|five|six|sept|seven|huit|eight|neuf|nine)\b", low)
        if not m:
            return None
        d = m.group(1)
        return d if d.isdigit() else _WORD2DIGIT.get(d)

    async def _press(digit: str):
        try:
            await ctx.room.local_participant.publish_dtmf(code=int(digit), digit=digit)
            logger.info("Mode OTP : touche %s pressee (DTMF envoye).", digit)
        except Exception:
            logger.exception("Echec envoi DTMF %s", digit)

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev):
        # Signal le plus fiable en téléphonie : si du texte est transcrit, l'utilisateur
        # a bel et bien parlé -> on annule le décompte de silence (et on reset le compteur).
        text = getattr(ev, "transcript", "") or ""
        if text.strip():
            _cancel_inactivity()
            _otp = os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
            is_final = getattr(ev, "is_final", True)
            if is_final:
                logger.info(f"Transcription (interlocuteur): {text}")
            elif _otp:
                # En mode capture OTP, logguer aussi les segments intermediaires :
                # en ecoute silencieuse, la finalisation peut ne jamais arriver.
                logger.info(f"Transcription OTP (interim): {text}")
            if _otp:
                _otp_buf.append(text.strip())
                del _otp_buf[:-6]
                joined = " ".join(_otp_buf)
                d = _extract_press_digit(joined)
                if d is not None and d not in _otp_pressed:
                    _otp_pressed.add(d)
                    logger.info("Mode OTP : instruction detectee dans '%s' -> touche %s", joined[-80:], d)
                    asyncio.create_task(_press(d))

    # --- Fin d'appel : on persiste TOUJOURS le transcript (appel sortant) -----
    # Le transcript est ajoute comme contenu de l'entree Journal UNIQUE de
    # l'appel (idempotence cote write_journal). append_journal_transcript cree
    # l'entree si aucune n'existe encore (ex. raccroche avant tout compte-rendu).
    async def _journal_fin_appel():
        try:
            if not is_outbound:
                return
            NL = chr(10)
            # Source principale : capture live (fiable même si raccroche brutal).
            lignes = list(live_transcript)
            if not lignes:
                # Repli : lecture de l'historique au shutdown (comportement historique).
                try:
                    for it in session.history.items:
                        role = getattr(it, "role", "")
                        if role not in ("user", "assistant"):
                            continue
                        txt = getattr(it, "text_content", None)
                        if txt is None:
                            c = getattr(it, "content", "")
                            if isinstance(c, list):
                                txt = " ".join(x for x in c if isinstance(x, str))
                            else:
                                txt = str(c)
                        if txt:
                            qui = "Agent" if role == "assistant" else "Interlocuteur"
                            lignes.append(qui + ": " + txt)
                except Exception:
                    logger.exception("transcript illisible au shutdown (repli history)")
            transcript = NL.join(lignes) if lignes else "(aucun echange capte)"
            await append_journal_transcript(transcript)
        except Exception:
            logger.exception("Echec compte-rendu auto au shutdown")
    
    _fin_appel_impl["fn"] = _journal_fin_appel
    
    # Ouverture conditionnelle.
    if is_outbound:
        # Appel sortant : l'ouverture suit la stratégie passée par lancer_appel
        # (instructions) si fournie, sinon une ouverture par défaut.
        ouverture = consignes_perso or (
            "Présente-toi brièvement comme l'assistante de Raphaël "
            "et annonce l'objet de ton appel."
        )
        # --- AMD (answering machine detection) : humain / SVI / repondeur ----
        # Classifie le decroche AVANT de derouler l'ouverture. Modeles LiveKit
        # Inference par defaut (gemini flash-lite + ink-whisper). En cas de
        # doute ou d'echec, on se comporte comme face a un humain.
        # AMD desactive par defaut (constat du 29/08/2026, trois appels muets).
        # Le plugin verrouille la parole de l'agent tant qu'il n'a pas conclu, et
        # il est concu pour demarrer AVANT la composition du numero. Ici c'est
        # voicecallmcp qui compose : l'agent arrive apres le decroche, l'AMD rate
        # le debut, tarde a conclure, et le verrou n'est pas toujours rendu — on
        # a vu l'interlocuteur repeter « allo, je n'entends rien » pendant que
        # l'agent restait muet. Le detecter correctement suppose de deplacer le
        # dial SIP dans l'agent : chantier, pas reglage. Mettre AMD_ENABLED=1
        # pour le reactiver (il fonctionne : un repondeur a bien ete classe
        # machine-vm et le message laisse).
        cat = "uncertain"
        _amd_on = os.environ.get("AMD_ENABLED", "").strip() in ("1", "true", "True")
        try:
            if not _amd_on:
                raise RuntimeError("AMD desactive (AMD_ENABLED absent)")
            # Seuils resserres (constat du 29/08/2026, en appel reel). Les
            # defauts du plugin (parole humaine 2,5 s, pas-de-parole 10 s,
            # plafond 20 s) supposent que l'AMD demarre AVANT la composition
            # du numero et entende toute la sonnerie. Ici c'est voicecallmcp
            # qui compose : l'agent arrive une fois la ligne ouverte, l'AMD
            # rate le debut et attend son plafond. Resultat observe : 35 s de
            # silence total pendant que l'interlocuteur repetait « allo ».
            # L'AMD met la parole en pause tant qu'il n'a pas conclu, donc son
            # plafond EST le silence initial de l'appel : on le borne a 3,5 s.
            amd_kwargs = {
                "wait_until_finished": False,  # sinon le plafond ne borne plus rien
                "detection_options": {
                    "human_speech_threshold": 1.0,   # un « allo » suffit
                    "human_silence_threshold": 0.4,
                    "no_speech_threshold": 2.5,
                    "timeout": 3.5,                  # plafond dur du silence initial
                },
            }
            if participant is not None:
                amd_kwargs["participant_identity"] = participant.identity
            async with AMD(session, **amd_kwargs) as detector:
                amd_result = await detector.execute()
            cat = getattr(getattr(amd_result, "category", None), "value", "uncertain") or "uncertain"
        except Exception as _amd_exc:
            if _amd_on:
                logger.exception("AMD indisponible, on continue comme avec un humain")
            else:
                logger.info("AMD : %s", _amd_exc)
        logger.info("AMD : %s", cat)
        if cat == "machine-vm":
            # Repondeur : laisser un message bref puis raccrocher.
            handle = session.generate_reply(
                instructions=(
                    "Tu es tombée sur un répondeur (aucun humain ne t'écoute). "
                    "Laisse UN message vocal bref et complet en une fois : qui tu es "
                    "(l'assistante de Raphaël Nicolle), l'objet de l'appel — "
                    + (objectif or "voir contexte") +
                    " — et invite à rappeler. Termine par au revoir. "
                    "Ne pose aucune question, n'attends aucune réponse."
                )
            )
            try:
                await handle.wait_for_playout()
            except Exception:
                logger.exception("Lecture du message repondeur interrompue")
            try:
                await write_journal_raw(
                    "Appel sortant : répondeur — message laissé",
                    "Objectif : " + (objectif or "?")[:300],
                    "info",
                )
            except Exception:
                logger.exception("Echec journal repondeur")
            await _hangup()
        elif cat == "machine-unavailable":
            # Messagerie pleine / non configuree : rien a faire.
            try:
                await write_journal_raw(
                    "Appel sortant : messagerie indisponible, aucun message possible",
                    "Objectif : " + (objectif or "?")[:300],
                    "erreur",
                )
            except Exception:
                logger.exception("Echec journal messagerie indisponible")
            await _hangup()
        else:
            # humain / incertain / SVI (la navigation SVI de l'AMD demarre
            # toute seule pour machine-ivr) : derouler l'ouverture normale.
            await session.generate_reply(instructions=ouverture)
    else:
        # Mode capture OTP : pour un appel de vérification (WhatsApp, etc.), on ne
        # dialogue PAS. L'agent reste silencieux et se contente d'écouter, ce qui
        # laisse le service dicter le code sans interruption. Les transcriptions
        # finales sont déjà journalisées (Transcription (interlocuteur): ...).
        otp_mode = os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
        otp_incl_raph = os.environ.get("OTP_CAPTURE_INCLUDE_RAPHAEL", "").strip() in ("1", "true", "True")
        otp_silence = otp_mode and (not is_raphael() or otp_incl_raph)
        if otp_silence:
            logger.info("Mode capture OTP actif : ecoute silencieuse (pas d'ouverture). include_raphael=%s", otp_incl_raph)
        else:
            # Appel entrant normal : accueil via session.say().
            if is_raphael():
                greeting = "Bonjour Raphaël, c'est Claude. Que puis-je faire pour vous ?"
            else:
                greeting = (
                    "Bonjour, vous êtes en communication avec l'assistant de Raphaël. "
                    "Puis-je savoir qui appelle ?"
                )
            await session.say(greeting, allow_interruptions=True)

        # Test isole du chemin DTMF : OTP_TEST_PRESS=7 -> presse 7 apres le decroche,
        # sans dependre du STT. A retirer une fois le chemin valide.
        _test_digit = os.environ.get("OTP_TEST_PRESS", "").strip()
        if _test_digit:
            async def _test_press():
                await asyncio.sleep(3)
                logger.info("TEST DTMF : envoi de la touche %s", _test_digit)
                await _press(_test_digit)
            asyncio.create_task(_test_press())


def prechauffer(proc: agents.JobProcess) -> None:
    """Ouvre la voie vers le LLM avant le premier appel, dans chaque process.

    Sans cela, la toute premiere requete d'un process fraichement demarre paie
    en une fois la connexion TLS vers le fournisseur ET la creation du cache de
    prompt : 4,4 s mesurees le 29/08/2026, contre 1,1 s ensuite. Sous la charge
    d'un vrai appel (STT, TTS et annulation de bruit en parallele), cela a
    depasse le timeout de 10 s de la session : deux « Request timed out » de
    suite, et un agent MUET pendant que Raphael repetait « allo ».

    Le prechauffage est volontairement discret : borne dans le temps, et une
    panne ici ne doit jamais empecher le worker de demarrer — un agent qui
    repond lentement au premier tour vaut mieux qu'un agent qui ne demarre pas.
    """
    if os.environ.get("PRECHAUFFAGE", "1").strip() in ("0", "false", "False"):
        logger.info("Prechauffage desactive (PRECHAUFFAGE=0)")
        return

    async def _chauffe() -> float | None:
        from livekit.agents import ChatContext

        llm = _make_llm(_resoudre_modele())
        ctx = ChatContext()
        ctx.add_message(role="system", content=get_system_prompt())
        ctx.add_message(role="user", content="Bonjour")
        debut = time.monotonic()
        async with llm.chat(chat_ctx=ctx, tools=TOOLS_RESTREINT) as flux:
            async for bout in flux:
                delta = getattr(bout, "delta", None)
                if delta and delta.content:
                    return time.monotonic() - debut
        return None

    try:
        duree = asyncio.run(asyncio.wait_for(_chauffe(), timeout=20))
        logger.info("Prechauffage LLM (%s) : premier mot en %.2f s",
                    _resoudre_modele(), duree or -1)
    except asyncio.TimeoutError:
        logger.warning("Prechauffage LLM abandonne (plus de 20 s), on demarre quand meme")
    except Exception:  # noqa: BLE001
        logger.warning("Prechauffage LLM impossible, on demarre quand meme", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prechauffer)
    )
