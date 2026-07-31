import asyncio
import json
import re
import logging
import os

import httpx

from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import (
    Agent, AgentSession, EndpointingOptions, RoomInputOptions, RoomOutputOptions,
    TurnHandlingOptions, inference, mcp, metrics,
)
from livekit.agents.voice.turn import InterruptionOptions, PreemptiveGenerationOptions
from livekit.plugins import anthropic, cartesia, deepgram, silero

from .system_prompt import SYSTEM_PROMPT
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
    _extra_no_prefill = ("claude-haiku-4-5",)
    _existing = tuple(_anthropic_llm._NO_PREFILL_PATTERNS)
    _merged = _existing + tuple(p for p in _extra_no_prefill if p not in _existing)
    _anthropic_llm._NO_PREFILL_PATTERNS = _merged
    logger.info("Patch prefill appliqué, modèles no-prefill: %s", _merged)
except Exception:  # noqa: BLE001
    logger.exception("Échec du patch _NO_PREFILL_PATTERNS")

# Délai de silence (secondes) avant relance puis raccrochage.
INACTIVITY_TIMEOUT = 10.0

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
]


async def _probe_mcp(url: str, headers: dict | None = None, timeout: float = 4.0) -> bool:
    """Vérifie qu'un serveur MCP répond avant de l'attacher à l'agent.

    Un serveur injoignable (DNS, refus de connexion, 4xx/5xx) ne doit JAMAIS
    faire planter toolset.setup(), ce qui rendrait l'agent muet après le
    message d'accueil. On ne garde le serveur que s'il répond avec un statut
    < 400.
    """
    probe_headers = {"Accept": "text/event-stream"}
    if headers:
        probe_headers.update(headers)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, headers=probe_headers) as resp:
                ok = resp.status_code < 400
                if not ok:
                    logger.warning("MCP %s répond %s, ignoré", url, resp.status_code)
                return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP %s injoignable (%s), ignoré", url, exc)
        return False


async def _build_mcp_servers() -> list:
    """Serveurs MCP self-hosted attachés UNIQUEMENT pour Raphaël identifié.

    Navigateur (human-browser) + WhatsApp. IMAP reste géré par les outils codés
    (get_unread_emails/send_email) pour éviter le doublon. Bitwarden est exclu.

    Chaque serveur est probé avant d'être attaché : un serveur injoignable est
    ignoré (avec log) au lieu de faire planter le setup des outils.
    """
    servers = []
    brmcp_url = os.environ.get("BRMCP_URL")
    brmcp_token = os.environ.get("BRMCP_TOKEN")
    if brmcp_url and brmcp_token:
        brmcp_headers = {"Authorization": f"Bearer {brmcp_token}"}
        if await _probe_mcp(brmcp_url, brmcp_headers):
            servers.append(
                mcp.MCPServerHTTP(
                    url=brmcp_url,
                    headers=brmcp_headers,
                    client_session_timeout_seconds=10,
                )
            )
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
    objectif = call_ctx.get("objectif", "")
    scenario = call_ctx.get("scenario", "")
    contexte_appel = call_ctx.get("contexte", "")
    consignes_perso = call_ctx.get("instructions", "")

    # Modele par appel : entrant => Sonnet (qualite, besoin imprevisible) ;
    # sortant => choix passe dans le metadata (defaut Haiku, sinon LLM_MODEL).
    _MODEL_MAP = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-4-6"}
    if is_outbound:
        _m = call_ctx.get("model") or os.environ.get("LLM_MODEL") or "sonnet"
        llm_model = _MODEL_MAP.get(_m, _m if _m.startswith("claude") else "claude-sonnet-4-6")
    else:
        llm_model = "claude-sonnet-4-6"
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

    prompt = SYSTEM_PROMPT
    if is_outbound:
        consignes = {
            "rdv": "Tu appelles pour prendre, décaler ou annuler un rendez-vous.",
            "rappel": "Tu appelles pour transmettre ou obtenir une information.",
            "message": "Tu appelles pour laisser un message à transmettre.",
            "relance": "Tu appelles pour relancer sur un sujet en attente.",
        }.get(scenario, "")
        prompt = SYSTEM_PROMPT + f"""

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
    mcp_servers = (await _build_mcp_servers()) if is_raphael() else []
    logger.info("Serveurs MCP attachés: %d (is_raphael=%s)", len(mcp_servers), is_raphael())

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
            else deepgram.STT(
                model="nova-3",
                base_url="https://api.eu.deepgram.com/v1/listen",
                language="fr",
                smart_format=True,
                punctuate=True,
            )
        ),
        llm=anthropic.LLM(
            # Modèle décidé par appel : entrant => Sonnet ; sortant => metadata.
            model=llm_model,
            temperature=0.7,
            caching="ephemeral",
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice=os.environ["CARTESIA_VOICE_ID"],
            language="fr",
        ),
        vad=silero.VAD.load(),
        # --- Gestion des tours de parole (API 1.6 : turn_handling unifié) -----
        # Remplace les anciens kwargs dépréciés (turn_detection / min|max_endpointing
        # _delay / preemptive_generation), qui se battaient entre eux.
        turn_handling=TurnHandlingOptions(
            # Détecteur audio v1 complet (Cloud). PAS d'unlikely_threshold forcé :
            # on garde les défauts calibrés par le serveur (les overrides dégradaient).
            turn_detection=inference.TurnDetector(version="v1"),
            # Endpointing dynamique : le délai s'adapte à la conversation au lieu
            # d'un seuil fixe rigide. Bornes raisonnables pour éviter les blancs.
            endpointing=(
                # Mode OTP : endpointing long et non-dynamique pour ne PAS clore
                # l'énoncé entre les chiffres dictés (sinon la 2e moitié est perdue).
                EndpointingOptions(mode="fixed", min_delay=2.5, max_delay=6.0)
                if os.environ.get("OTP_CAPTURE_MODE", "").strip() in ("1", "true", "True")
                else EndpointingOptions(mode="dynamic", min_delay=0.4, max_delay=1.5)
            ),
            # Interruptions ADAPTATIVES (barge-in propre) : c'était désactivé par
            # défaut en prod, d'où l'impression de blancs/coupures. On l'active.
            interruption=InterruptionOptions(enabled=True, mode="adaptive"),
            # Génération préemptive : le LLM démarre pendant que l'utilisateur finit.
            preemptive_generation=PreemptiveGenerationOptions(enabled=True),
        ),
        # Serveurs MCP (navigateur + WhatsApp), uniquement pour Raphaël identifié.
        mcp_servers=mcp_servers,
    )

    # --- Capture live du transcript (fix bug 02/07/2026) ------------------
    # session.history.items lu au shutdown peut manquer le dernier tour si
    # l'appel se coupe brutalement (raccroche interlocuteur) avant que le
    # tour soit "committed". conversation_item_added est émis de façon
    # synchrone dès qu'un item (user OU assistant) est ajouté au contexte :
    # on l'utilise comme source de vérité principale pour le transcript.
    live_transcript: list[str] = []

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
        except Exception:
            logger.exception("Erreur capture live transcript (conversation_item_added)")

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=prompt,
            tools=TOOLS,
        ),
        # noise_cancellation désactivé : le plugin BVC n'est pas installé et le
        # laisser actif peut bloquer silencieusement la publication audio.
        # delete_room_on_close ferme la room (et coupe la ligne) à la fin.
        room_input_options=RoomInputOptions(delete_room_on_close=True),
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

    # --- Mesure de latence : loguer les métriques de chaque tour ----------
    @session.on("metrics_collected")
    def _on_metrics(ev):
        try:
            metrics.log_metrics(ev.metrics)
        except Exception:  # noqa: BLE001
            logger.debug("metrics: %s", getattr(ev, "metrics", None))

    async def _hangup() -> None:
        try:
            # Attendre la fin de la parole en cours avant de couper la ligne.
            try:
                await session.wait_for_idle()
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2.0)
            await asyncio.sleep(0.3)  # marge réseau dernier paquet audio
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:  # noqa: BLE001
            logger.exception("Échec du raccrochage automatique")

    async def _inactivity_watch() -> None:
        nonlocal relance_count
        try:
            await asyncio.sleep(INACTIVITY_TIMEOUT)
        except asyncio.CancelledError:
            return
        # Ne relancer que si l'agent ne parle pas déjà (sécurité).
        relance_count += 1
        if relance_count == 1:
            await session.say("Vous êtes toujours là ?", allow_interruptions=True)
            # Réarmer pour laisser une 2e chance APRÈS la relance.
            _arm_inactivity()
        else:
            await session.say(
                "Je vais raccrocher, n'hésitez pas à me rappeler. Au revoir.",
                allow_interruptions=False,
            )
            await _hangup()

    def _arm_inactivity() -> None:
        nonlocal inactivity_task
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
        new = getattr(ev, "new_state", None)
        if new == "speaking":
            # L'agent parle : ce n'est pas un silence, on suspend le décompte.
            if inactivity_task and not inactivity_task.done():
                inactivity_task.cancel()
        elif new == "listening":
            # L'agent a fini de parler et attend l'utilisateur : (re)lancer le décompte.
            # (couvre le cas où l'utilisateur était déjà silencieux, donc aucun
            #  user_state_changed ne se déclenche après la réponse de l'agent)
            _arm_inactivity()

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
