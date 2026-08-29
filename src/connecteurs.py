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

# --- Registre : nom -> (palier requis, resume, outils autorises) ------------
# Les resumes sont ce que le LLM lit dans lister_connecteurs : courts, en
# francais, orientes usage telephonique.
REGISTRE: dict[str, dict] = {
    "calibre": {
        "palier": 1,
        "resume": "Bibliotheque de livres : recherche, metadonnees, parutions.",
        "outils": ["chercher", "lister_series", "parutions_serie", "manques",
                   "stock_par_serie"],
    },
    "photos": {
        "palier": 1,
        "resume": "Bibliotheque photo (PhotoPrism) : recherche par personne, lieu, date.",
        "outils": ["chercher", "photo", "apercu", "albums", "etiquettes",
                   "personnes", "statistiques"],
    },
    "beeper": {
        "palier": 1,
        "resume": "Messageries (WhatsApp, SMS, Signal...) : lire les conversations recentes.",
        "outils": ["search_chats", "list_messages", "get_chat", "search_messages",
                   "search", "get_accounts"],
    },
    "qonto": {
        "palier": 2,
        "resume": "Comptes bancaires professionnels : soldes, transactions, factures.",
        "outils": ["qonto_consolidated_balances", "qonto_list_transactions",
                   "qonto_get_transaction", "qonto_list_organizations",
                   "qonto_list_supplier_invoices", "qonto_list_client_invoices",
                   "qonto_list_statements"],
    },
    "crypto": {
        "palier": 2,
        "resume": "Portefeuille crypto : valorisation, soldes, positions. Lecture seule.",
        "outils": ["get_portfolio", "get_balances", "get_crypto_prices",
                   "get_eth_tokens", "get_bch_balances", "get_defi_positions",
                   "get_history", "list_wallets", "safe_info"],
    },
    "digifinex": {
        "palier": 2,
        "resume": "Compte d'echange DigiFinex : soldes spot, ordres, historique.",
        "outils": ["digifinex_spot_assets", "digifinex_ticker",
                   "digifinex_my_trades", "digifinex_open_orders",
                   "digifinex_order_history", "digifinex_financelog"],
    },
    "fichiers": {
        "palier": 2,
        "resume": "Fichiers du serveur : lister, lire un document. Lecture seule.",
        "outils": ["list_roots", "list_dir", "stat", "read_text", "read_document"],
    },
}

# Ce que la passerelle refuse toujours de relayer, meme si un jour la liste
# blanche d'un connecteur s'elargit par erreur. Ceinture et bretelles.
MOTIFS_INTERDITS = ("delete", "supprim", "send", "envoy", "write", "ecrire", "push",
                    "upload", "transfer", "virement", "withdraw", "broadcast",
                    "propose", "sign", "bash", "shell", "exec")


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
    cfg = REGISTRE.get(nom_connecteur)
    if not cfg:
        return False, f"connecteur inconnu : {nom_connecteur}"
    bas = outil.lower()
    for motif in MOTIFS_INTERDITS:
        if motif in bas:
            return False, (
                f"l'outil « {outil} » ressemble a une action d'ecriture ou de "
                "suppression : refuse par telephone"
            )
    # Le nom peut arriver prefixe par le serveur (ex. qonto_qonto_list_...).
    court = outil.split("_", 1)[1] if outil.startswith(nom_connecteur + "_") else outil
    if court not in cfg["outils"] and outil not in cfg["outils"]:
        return False, (
            f"outil « {outil} » hors de la liste autorisee pour {nom_connecteur}. "
            f"Autorises : {', '.join(cfg['outils'])}"
        )
    return True, court


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
