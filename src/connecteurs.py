"""Passerelle vers les connecteurs MCP, avec habilitation par palier.

POURQUOI UNE PASSERELLE PLUTOT QUE DES SERVEURS MCP ATTACHES
------------------------------------------------------------
Attacher les connecteurs a la session LiveKit expose TOUS leurs outils au
LLM : les schemas partent dans le prompt de CHAQUE tour. Mesure du 29/08/2026
sur cet agent : 14 outils codes + 2 serveurs MCP = ~4000 tokens de schemas
pour ~1960 tokens de prompt systeme. Ajouter huit connecteurs (~150 outils)
aurait triple le prompt et donc le TTFT — l'inverse de ce qu'on cherche.

Ici, trois outils seulement entrent dans le prompt (lister_connecteurs,
lister_outils, utiliser_connecteur) et le catalogue se decouvre a la demande.
Le cout en latence est constant quel que soit le nombre de connecteurs.

HABILITATION : TROIS PALIERS
----------------------------
Le caller ID SIP est falsifiable ; c'est le point faible assume de cet agent.
On ne lui fait donc porter que des lectures anodines, et tout ce qui touche a
l'argent, aux emails ou aux fichiers exige le mot de passe vocal.

  palier 0 — appel sortant, ou appelant inconnu : AUCUN connecteur.
  palier 1 — caller ID = Raphael : lectures anodines (memoire, calibre,
             photos, beeper en lecture).
  palier 2 — mot de passe vocal verifie (outil verifier_identite) :
             + qonto, crypto, digifinex, fichiers (lecture).

Jamais accessibles par telephone, quel que soit le palier : shell, suppression,
virements, ecritures de fichiers. La liste blanche d'outils par connecteur est
la seule autorite — un outil non liste n'est pas appelable, meme s'il existe
cote serveur (protege des nouveaux outils d'ecriture ajoutes en amont).
"""
from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger("voice-agent.connecteurs")

TIMEOUT = 20.0

# --- Registre : nom -> (palier requis, resume) -----------------------------
# Les resumes sont ce que le LLM lit dans lister_connecteurs : courts, en
# francais, orientes usage telephonique. Les outils ne sont PAS listes ici :
# a ~260 outils sur 16 serveurs, une liste blanche nominative serait fausse
# des la prochaine mise a jour d'un connecteur. La regle est structurelle et
# vaut pour tous : lecture autorisee, ecriture refusee (voir _LECTURE /
# _ECRITURE plus bas), avec le meme resultat — un outil ajoute en amont est
# refuse par defaut tant qu'il ne ressemble pas a une lecture.
REGISTRE: dict[str, dict] = {
    # --- Palier 1 : caller ID de Raphael. Rien de sensible ici. -------------
    "calibre": {"palier": 1, "resume": "Bibliotheque de livres : series, recherche, parutions, manques."},
    "photos": {"palier": 1, "resume": "Bibliotheque photo : recherche, personnes, albums, statistiques."},
    "beeper": {"palier": 1, "resume": "Messageries (WhatsApp, SMS, Signal...) : lire conversations et messages."},
    "gcal": {"palier": 1, "resume": "Agenda Google : calendriers, evenements, disponibilites."},
    "sched": {"palier": 1, "resume": "Taches planifiees du serveur : lister ce qui tourne."},
    # --- Palier 2 : mot de passe vocal. Argent, documents, correspondance. --
    "qonto": {"palier": 2, "resume": "Banque Qonto : soldes, transactions, factures, releves."},
    "amina": {"palier": 2, "resume": "Banque AMINA (crypto, Zug) : comptes, soldes, transactions."},
    "crypto": {"palier": 2, "resume": "Portefeuille crypto : valorisation, soldes, positions DeFi, historique."},
    "digifinex": {"palier": 2, "resume": "Echange DigiFinex : soldes spot, ordres, trades, historique."},
    "pennylane": {"palier": 2, "resume": "Comptabilite Pennylane : factures, ecritures, comptes, clients."},
    "fichiers": {"palier": 2, "resume": "Fichiers du serveur : lister, lire un document."},
    "rclone": {"palier": 2, "resume": "Stockages distants (Drive, S3...) : lister, lire, chercher."},
    "imap2": {"palier": 2, "resume": "Emails : chercher, lire, fils de discussion, etat des boites."},
    "notion": {"palier": 2, "resume": "Notion : rechercher et lire pages et bases."},
    "wp": {"palier": 2, "resume": "Sites WordPress : articles, pages, medias, commentaires (lecture)."},
    "twilio2": {"palier": 2, "resume": "Telephonie Twilio : historique des appels et des SMS recus."},
}

# --- Lecture seule : deux filtres qui doivent tomber d'accord ---------------
# 1. l'outil doit RESSEMBLER a une lecture (_LECTURE), sinon il est refuse ;
# 2. il ne doit contenir AUCUN motif d'ecriture (_ECRITURE), meme s'il a passe
#    le premier filtre — « imap_bulk_delete_by_search » contient « search ».
# Refus par defaut : ce qui n'est pas reconnu comme une lecture ne passe pas.
_LECTURE = (
    "get", "list", "lister", "read", "lire", "search", "cherch", "recherch",
    "find", "trouv", "query", "fetch", "retrieve", "stat", "info", "whoami",
    "self", "balance", "portfolio", "ticker", "history", "historique",
    "consolidated", "apercu", "photo", "albums", "etiquettes", "personnes",
    "parutions", "manques", "series", "freebusy", "current-time", "status",
    "pending", "colors", "assets", "trades", "orders", "log", "chat",
    "messages", "accounts", "abilities", "site", "theme", "backups",
    "export", "decode", "download", "pull", "stock", "transcri", "unread",
    "posts", "pages", "media", "comments", "labels", "memberships",
)
_ECRITURE = (
    "delete", "suppr", "remove", "retir", "del_", "purge", "clear",
    "send", "envoy", "reply", "repond", "respond", "forward", "post-page",
    "post_page", "create_post", "create-post",
    "creer", "cree", "create", "add", "ajout", "insert", "new_",
    "update", "patch", "put", "modif", "edit", "set_", "definir", "rename",
    "move", "deplac", "archive", "restore", "save", "draft", "upload",
    "push", "write", "ecrire", "televerser", "integrer", "indexer",
    "marqu", "mark", "approve", "whitelist", "build", "sign", "broadcast",
    "propose", "discard", "revoke", "transfer", "virement", "withdraw",
    "exec", "run", "bash", "shell", "manage", "moderate", "connect",
    "disconnect", "hangup", "speak", "dtmf", "make_call", "focus", "reminder",
)


def _env(nom: str, suffixe: str) -> str:
    return os.environ.get(f"CX_{nom.upper()}_{suffixe}", "").strip()


def connecteurs_disponibles(palier: int) -> dict[str, dict]:
    """Connecteurs configures (URL + token presents) et ouverts a ce palier."""
    return {
        nom: cfg
        for nom, cfg in REGISTRE.items()
        if cfg["palier"] <= palier and _env(nom, "URL") and _env(nom, "TOKEN")
    }


def outil_autorise(nom_connecteur: str, outil: str) -> tuple[bool, str]:
    """Un outil passe s'il ressemble a une lecture et a rien qui ecrive."""
    if nom_connecteur not in REGISTRE:
        return False, f"connecteur inconnu : {nom_connecteur}"
    bas = outil.lower()
    for motif in _ECRITURE:
        if motif in bas:
            return False, (
                f"« {outil} » modifie, envoie ou supprime quelque chose : "
                "refuse par telephone, quel que soit l'appelant"
            )
    if not any(motif in bas for motif in _LECTURE):
        return False, (
            f"« {outil} » n'est pas reconnu comme une lecture ; par prudence il "
            "est refuse. Utilise lister_outils pour voir ce qui est disponible."
        )
    return True, outil


async def _session_mcp(client: httpx.AsyncClient, url: str, headers: dict) -> dict:
    """Ouvre une session MCP et rend les en-tetes a rejouer pour la suite.

    Certains proxys (memoire, crypto, digifinex) refusent un tools/list nu avec
    -32602 ou repondent un corps vide : ils exigent le handshake complet
    initialize -> notifications/initialized, et rattachent la suite a l'identifiant
    de session rendu dans l'en-tete Mcp-Session-Id. D'autres (qonto, photos)
    acceptent l'appel direct. On fait le handshake pour tout le monde : c'est un
    aller-retour de plus, mais il vaut mieux que la moitie des connecteurs tombe.
    """
    init = {
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-vocal", "version": "1.0"},
        },
    }
    r = await client.post(url, headers=headers, json=init)
    r.raise_for_status()
    sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
    suite = dict(headers)
    if sid:
        suite["Mcp-Session-Id"] = sid
    await client.post(url, headers=suite, json={
        "jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    return suite


def _corps_json(texte: str) -> dict:
    """Lit une reponse JSON-RPC, que le transport soit du JSON nu ou du SSE."""
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if ligne.startswith("data:"):
            texte = ligne[5:].strip()
            break
    if not texte.strip():
        raise RuntimeError("reponse vide du connecteur")
    return json.loads(texte)


async def _appel_mcp(nom: str, methode: str, params: dict) -> dict:
    """Un appel outil sur le connecteur, handshake MCP compris."""
    url, token = _env(nom, "URL"), _env(nom, "TOKEN")
    if not url or not token:
        raise RuntimeError(f"connecteur {nom} non configure")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        suite = await _session_mcp(client, url, headers)
        r = await client.post(url, headers=suite, json={
            "jsonrpc": "2.0", "id": 1, "method": methode, "params": params})
        r.raise_for_status()
        data = _corps_json(r.text)
    if "error" in data:
        raise RuntimeError(str(data["error"])[:200])
    return data.get("result", {})


async def lister_outils_distants(nom: str) -> list[dict]:
    res = await _appel_mcp(nom, "tools/list", {})
    return res.get("tools", []) or []


async def appeler_outil(nom: str, outil: str, arguments: dict) -> str:
    res = await _appel_mcp(nom, "tools/call", {"name": outil, "arguments": arguments})
    morceaux = []
    for bloc in res.get("content", []) or []:
        if bloc.get("type") == "text":
            morceaux.append(bloc.get("text", ""))
    return "\n".join(morceaux) if morceaux else json.dumps(res)[:2000]
