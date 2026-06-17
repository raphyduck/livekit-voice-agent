import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("voice-agent.mcp")


class MCPClient:
    """Client HTTP pour serveurs MCP avec authentification Bearer.

    Compatible avec les serveurs MCP "Streamable HTTP" qui peuvent répondre
    soit en JSON classique, soit en flux SSE (text/event-stream).
    """

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict:
        """Extrait le payload JSON-RPC d'une réponse JSON ou SSE."""
        content_type = response.headers.get("content-type", "")
        text = response.text

        if "text/event-stream" in content_type:
            # Concatène les lignes "data:" et garde le dernier message valide.
            last: dict = {}
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    last = json.loads(payload)
                except json.JSONDecodeError:
                    continue
            return last

        return response.json()

    async def list_tools(self) -> list[dict]:
        """Retourne la liste des outils disponibles sur ce serveur MCP."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                self.base_url,
                headers=self.headers,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=10.0,
            )
            r.raise_for_status()
            data = self._parse_response(r)
            return data.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Appelle un outil MCP et retourne le résultat sous forme de texte."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                self.base_url,
                headers=self.headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
                timeout=30.0,
            )
            r.raise_for_status()
            data = self._parse_response(r)

            if "error" in data:
                err = data["error"]
                logger.warning("Erreur MCP sur %s: %s", tool_name, err)
                return f"Erreur de l'outil : {err.get('message', 'inconnue')}"

            result = data.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                joined = "\n".join(t for t in texts if t)
                return joined or str(result)
            return str(result)
