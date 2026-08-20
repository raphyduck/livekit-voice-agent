"""System prompt de l'agent vocal, tire du cerveau (montage RO /brain).

Source de verite unique : la note agents/agent-vocal.md du depot memoire.
Frontmatter retire, {{include: ...}} resolus, intitules documentaires retires.
Cache de la derniere version valide dans /data/prompt-cache.md ; sans cerveau
ni cache, echec explicite au demarrage plutot qu'un agent au prompt vide.
Conception : divers/architecture-prompts-agents-cerveau-source-unique.md.
"""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("system_prompt")

BRAIN_DIR = os.environ.get("BRAIN_DIR", "/brain")
PROMPT_NOTE = os.environ.get("PROMPT_NOTE", "agents/agent-vocal.md")
CACHE_PATH = os.environ.get("PROMPT_CACHE", "/data/prompt-cache.md")

_INCLUDE_RE = re.compile(r"^\s*\{\{include:\s*([^}]+?)\s*\}\}\s*$", re.M)
_HEADING_DOC_RE = re.compile(r"^##\s+Instruction\b[^\n]*\n", re.M)


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def _read(rel_path: str) -> str:
    with open(os.path.join(BRAIN_DIR, rel_path), encoding="utf-8") as fh:
        return fh.read()


def _resolve_includes(text: str, depth: int = 0) -> str:
    if depth > 3:
        raise ValueError("inclusions imbriquees au-dela de 3 niveaux")

    def _repl(match: "re.Match[str]") -> str:
        return _resolve_includes(_strip_frontmatter(_read(match.group(1))), depth + 1).strip() + "\n"

    return _INCLUDE_RE.sub(_repl, text)


def get_system_prompt() -> str:
    """Charge le prompt depuis le cerveau, avec repli sur le dernier cache valide."""
    try:
        body = _resolve_includes(_strip_frontmatter(_read(PROMPT_NOTE)))
        body = _HEADING_DOC_RE.sub("", body).strip()
        if len(body) < 500:
            raise ValueError(f"prompt suspect ({len(body)} caracteres)")
        try:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as fh:
                fh.write(body)
        except OSError as exc:
            log.warning("cache prompt non ecrit: %s", exc)
        return body
    except Exception as exc:
        log.warning("cerveau illisible (%s), repli sur le cache %s", exc, CACHE_PATH)
        with open(CACHE_PATH, encoding="utf-8") as fh:
            return fh.read()


# Compat : certains imports historiques attendent la constante. Evaluee au
# demarrage du worker ; les sessions d'appel repassent par get_system_prompt().
SYSTEM_PROMPT = get_system_prompt()
