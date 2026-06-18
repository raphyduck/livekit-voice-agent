import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import Agent, AgentSession, RoomInputOptions, RoomOutputOptions, metrics
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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
    verifier_identite,
    write_journal,
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
    get_current_datetime,
    end_call,
    verifier_identite,
]


async def entrypoint(ctx: agents.JobContext):
    logger.info("Agent démarré pour la room: %s", ctx.room.name)
    await ctx.connect()

    # --- Identification de l'appelant (caller ID, authentification faible) ----
    # NB : le caller ID SIP est spoofable et un mot de passe parlé est faible ;
    # c'est un compromis assumé pour un usage perso, pas une sécurité forte.
    reset_identity(False)  # repartir non identifié à chaque appel
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
- C'est TOI qui appelles, pas l'inverse. Présente-toi brièvement et explique l'objet de ton appel.
- {consignes}
- Objectif précis : {objectif}
- Informations utiles : {contexte_appel or "aucune"}
- Mène la conversation vers cet objectif, poliment et efficacement.
- À la fin de l'appel, AVANT de raccrocher, rédige un compte-rendu concis avec write_journal
  (type "action") résumant : qui tu as appelé, ce qui a été dit, le résultat obtenu.
"""

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="fr",
            smart_format=True,
            punctuate=True,
        ),
        llm=anthropic.LLM(
            # Modèle configurable via .env (LLM_MODEL). Défaut: Sonnet.
            # Pour tester Haiku (plus rapide) : LLM_MODEL=claude-haiku-4-5
            model=os.environ.get("LLM_MODEL", "claude-haiku-4-5"),
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            model="sonic-2",
            voice=os.environ["CARTESIA_VOICE_ID"],
            language="fr",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        # --- Optimisations latence ---
        # Réduire le délai avant de considérer que l'utilisateur a fini de parler
        # (EOU mesuré ~1.4s par défaut). Plus court = réponse plus rapide, au prix
        # d'un risque accru de couper si l'utilisateur fait une longue pause.
        min_endpointing_delay=0.4,
        max_endpointing_delay=3.0,
        # Laisser le LLM commencer à générer pendant que l'utilisateur finit :
        # gros gain de latence perçue.
        preemptive_generation=True,
    )

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

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev):
        # Signal le plus fiable en téléphonie : si du texte est transcrit, l'utilisateur
        # a bel et bien parlé -> on annule le décompte de silence (et on reset le compteur).
        text = getattr(ev, "transcript", "") or ""
        if text.strip():
            _cancel_inactivity()

    # Ouverture conditionnelle.
    if is_outbound:
        # Appel sortant : c'est l'agent qui ouvre. On laisse le LLM générer
        # l'ouverture selon l'objectif plutôt qu'une phrase figée.
        await session.generate_reply(
            instructions="Présente-toi brièvement comme l'assistante de Raphaël "
            "et annonce l'objet de ton appel."
        )
    else:
        # Appel entrant : accueil via session.say() (plus fiable que generate_reply,
        # ne dépend pas du LLM et teste directement le chemin TTS → track audio).
        if is_raphael():
            greeting = "Bonjour Raphaël, c'est Claude. Que puis-je faire pour vous ?"
        else:
            greeting = (
                "Bonjour, vous êtes en communication avec l'assistant de Raphaël. "
                "Puis-je savoir qui appelle ?"
            )
        await session.say(greeting, allow_interruptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
