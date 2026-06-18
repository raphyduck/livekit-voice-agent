import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import agents, api
from livekit.agents import Agent, AgentSession, RoomInputOptions, RoomOutputOptions
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
    read_brain,
    send_email,
    send_sms,
    write_journal,
)

load_dotenv()
logger = logging.getLogger("voice-agent")

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
]


async def entrypoint(ctx: agents.JobContext):
    logger.info("Agent démarré pour la room: %s", ctx.room.name)
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
            model="sonic-2",
            voice=os.environ["CARTESIA_VOICE_ID"],
            language="fr",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=SYSTEM_PROMPT,
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
        relance_count += 1
        if relance_count == 1:
            # Le retour en état "listening" ré-armera automatiquement le minuteur.
            await session.say("Vous êtes toujours là ?", allow_interruptions=True)
        else:
            await session.say(
                "Je vais raccrocher, n'hésitez pas à me rappeler. Au revoir.",
                allow_interruptions=False,
            )
            await _hangup()

    def _arm_inactivity() -> None:
        nonlocal inactivity_task
        current = asyncio.current_task()
        # Ne jamais s'auto-annuler : on ignore la tâche courante.
        if inactivity_task and inactivity_task is not current and not inactivity_task.done():
            inactivity_task.cancel()
        inactivity_task = asyncio.create_task(_inactivity_watch())

    def _cancel_inactivity() -> None:
        nonlocal inactivity_task, relance_count
        relance_count = 0
        if inactivity_task and not inactivity_task.done():
            inactivity_task.cancel()
        inactivity_task = None

    @session.on("user_state_changed")
    def _on_user_state(ev):
        if getattr(ev, "new_state", None) == "speaking":
            _cancel_inactivity()

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        if getattr(ev, "new_state", None) == "listening":
            _arm_inactivity()

    # Premier message via session.say() : plus fiable que generate_reply() car il
    # ne dépend pas du LLM et teste directement le chemin TTS → track audio.
    await session.say(
        "Bonjour, c'est Claude, l'assistant de Raphaël. Que puis-je faire pour vous ?",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
