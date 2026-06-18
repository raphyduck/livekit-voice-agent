# AGENT_IMPROVEMENTS.md — Raccrochage + relance sur silence

Cible : `~/docker_images/livekit-voice-agent` (livekit-agents 1.6.0, Python).
Fichiers concernés : `src/agent.py`, `src/tools.py`, `src/system_prompt.py`.
Après modif : commit + push (remote SSH `git@github.com:raphyduck/livekit-voice-agent.git`), puis sur le serveur `cd ~/docker_images/livekit-voice-agent && docker compose up -d --build` (REBUILD obligatoire, le code est copié dans l'image).

## Objectif 1 — L'agent sait raccrocher

Ajouter un outil `end_call` que le LLM appelle quand la conversation est terminée
(l'interlocuteur dit au revoir, ou la tâche est finie).

### Dans `src/tools.py`
```python
from livekit.agents import function_tool, get_job_context
from livekit import api
import asyncio

@function_tool()
async def end_call() -> str:
    """Raccroche et met fin a l'appel telephonique en cours.
    A utiliser quand l'interlocuteur dit au revoir, que la conversation
    est clairement terminee, ou qu'il n'y a plus rien a faire."""
    ctx = get_job_context()
    await asyncio.sleep(0.5)  # laisser le dernier TTS se jouer
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    return "Appel termine."
```
VERIFIER dans la version installee que `ctx.api.room.delete_room` est le bon chemin.
Sinon, construire `api.LiveKitAPI()` depuis les env LIVEKIT_URL/KEY/SECRET et appeler
`.room.delete_room(...)`. Tester l'attribut reel avant de figer.

### Dans `src/agent.py`
- Importer `end_call`, l'ajouter a la liste `TOOLS`.
- Dans `room_input_options=RoomInputOptions(...)`, ajouter `delete_room_on_close=True`.

### Dans `src/system_prompt.py`
Ajouter :
```
RACCROCHAGE :
- Quand l'interlocuteur dit au revoir ou que la conversation est terminee,
  dis une formule de politesse courte PUIS utilise l'outil end_call.
- N'utilise JAMAIS end_call sans avoir dit au revoir d'abord.
```

## Objectif 2 — Relance quand l'interlocuteur reste silencieux

Comportement : si l'utilisateur ne dit rien ~10s apres que l'agent a fini de parler,
relancer ("Vous etes toujours la ?"). Apres une 2e periode de silence (~10s),
raccrocher poliment.

### Mecanisme (livekit-agents 1.6.0)
`AgentSession` emet `UserStateChangedEvent` et `AgentStateChangedEvent` et expose
`session.user_state` / `session.agent_state`. Armer un timer d'inactivite.

Dans `entrypoint()` APRES `session.start(...)` :
```python
import asyncio

inactivity_task = None
relance_count = 0

async def _inactivity_watch():
    nonlocal relance_count
    try:
        await asyncio.sleep(10)
        relance_count += 1
        if relance_count == 1:
            await session.say("Vous etes toujours la ?", allow_interruptions=True)
            _arm_inactivity()
        else:
            await session.say("Je vais raccrocher, n'hesitez pas a me rappeler. Au revoir.",
                              allow_interruptions=False)
            ctx2 = get_job_context()
            await ctx2.api.room.delete_room(api.DeleteRoomRequest(room=ctx2.room.name))
    except asyncio.CancelledError:
        pass

def _arm_inactivity():
    nonlocal inactivity_task
    if inactivity_task and not inactivity_task.done():
        inactivity_task.cancel()
    inactivity_task = asyncio.create_task(_inactivity_watch())

def _cancel_inactivity():
    nonlocal relance_count
    relance_count = 0
    if inactivity_task and not inactivity_task.done():
        inactivity_task.cancel()

@session.on("user_state_changed")
def _on_user_state(ev):
    if getattr(ev, "new_state", None) == "speaking":
        _cancel_inactivity()

@session.on("agent_state_changed")
def _on_agent_state(ev):
    if getattr(ev, "new_state", None) == "listening":
        _arm_inactivity()
```

POINTS A VERIFIER (ne pas supposer, confirmer dans la version installee) :
- Noms d'events exacts : "user_state_changed" / "agent_state_changed" vs classes
  UserStateChangedEvent / AgentStateChangedEvent. Logger l'event recu d'abord.
- Valeurs de `new_state` : confirmer "speaking" / "listening" / "away".
- Signature du callback `@session.on(...)` (1 argument event ?).
- Callbacks synchrones : lancer les coroutines via asyncio.create_task.
- Si un mecanisme d'inactivite natif existe (timeout configurable sur la session
  ou RoomInputOptions), le preferer au timer manuel.

## Tests apres deploiement (REBUILD obligatoire)
1. `docker compose up -d --build` puis verifier `registered worker`.
2. Appel entrant +13185598381 :
   - "au revoir" -> l'agent salue puis raccroche (ligne coupee).
   - silence 10s -> "Vous etes toujours la ?".
   - silence encore 10s -> adieu + raccrochage.
3. Logs : pas d'erreur delete_room, pas de double timer.

## Ne pas toucher
STT nova-3 fr, TTS Cartesia sonic-2, LLM Anthropic, RoomOutputOptions(audio_enabled=True).
Garder logging DEBUG pour l'instant.
