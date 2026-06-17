import logging
import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from .system_prompt import SYSTEM_PROMPT
from .tools import (
    add_task,
    create_calendar_event,
    get_calendar_events,
    get_current_datetime,
    get_today_tasks,
    get_unread_emails,
    search_notion,
    send_email,
    send_sms,
)

load_dotenv()
logger = logging.getLogger("voice-agent")

TOOLS = [
    get_calendar_events,
    create_calendar_event,
    get_unread_emails,
    send_email,
    get_today_tasks,
    add_task,
    search_notion,
    send_sms,
    get_current_datetime,
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
            model="sonic-multilingual",
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
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Salue l'appelant en français, présente-toi comme Claude "
        "l'assistant de Raphaël, et demande ce que tu peux faire."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
