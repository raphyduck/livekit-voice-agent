# BRAIN_TOOLS.md — Acces complet au brain Notion pour l'agent vocal

Cible : ~/docker_images/livekit-voice-agent (livekit-agents 1.6.0).
Fichiers : src/tools.py (ajouter 2 tools), src/agent.py (les enregistrer), src/system_prompt.py (documenter).
Apres : commit + push (remote SSH deja configure : git@github.com:raphyduck/livekit-voice-agent.git), puis sur le serveur `docker compose up -d --build` (REBUILD obligatoire).

## Contexte
Le tool `search_notion` actuel ne renvoie QUE les titres des 3 premiers resultats (inutilisable
pour vraiment consulter le brain). On le remplace par un acces riche en LECTURE et on ajoute
l'ECRITURE dans le Journal. Auth via l'API Notion REST (token deja present dans .env :
NOTION_API_KEY, et NOTION_BRAIN_PAGE_ID). Notion-Version: 2022-06-28.

IDs des bases du brain (parent page AI brain = valeur de NOTION_BRAIN_PAGE_ID) :
- Profil       : 7df5cd62-8e47-40b9-ba58-e2de3c8a6be2  (faits stables ; lecture seule pour l'agent)
- Agents       : f611ebb1-a451-4759-87f9-15fe64ac7ac6
- Journal      : 1781b732-9e14-42f7-9c61-ab63e3f8ff0d  (append-only ; l'agent ECRIT ici)
- Items ouverts: a2fae549-6e30-4d95-98cb-8ab24516bb4f

## Tool 1 — remplacer `search_notion` par `read_brain` (lecture riche)

Comportement : chercher dans le brain, puis pour les meilleurs resultats RECUPERER LE CONTENU
(pas juste le titre). Utiliser l'endpoint /v1/search puis, pour chaque page trouvee (max 3),
recuperer ses blocks via /v1/blocks/{page_id}/children et concatener le texte.

```python
@function_tool()
async def read_brain(query: str) -> str:
    """Consulte le cerveau (brain) Notion de Raphael : profil, agents, journal, items ouverts.
    Renvoie le contenu des entrees les plus pertinentes, pas seulement les titres.

    Args:
        query: Termes a rechercher (sujet, nom d'agent, infra, etc.).
    """
    import os, httpx
    headers = {
        "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=READ_TIMEOUT) as client:
            r = await client.post("https://api.notion.com/v1/search",
                                  headers=headers, json={"query": query, "page_size": 3})
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                return "Aucun resultat dans le brain."
            out = []
            for page in results[:3]:
                title = _extract_notion_title(page)
                pid = page.get("id")
                # Recuperer le contenu (blocks) de la page
                text_parts = []
                try:
                    br = await client.get(
                        f"https://api.notion.com/v1/blocks/{pid}/children?page_size=20",
                        headers=headers)
                    br.raise_for_status()
                    for blk in br.json().get("results", []):
                        t = blk.get(blk.get("type"), {})
                        for rt in t.get("rich_text", []):
                            text_parts.append(rt.get("plain_text", ""))
                except Exception:
                    pass
                # Inclure aussi les valeurs des proprietes texte/select de la page
                props = page.get("properties", {})
                prop_bits = []
                for name, p in props.items():
                    ptype = p.get("type")
                    if ptype == "rich_text":
                        prop_bits.append(" ".join(rt.get("plain_text","") for rt in p.get("rich_text",[])))
                    elif ptype == "select" and p.get("select"):
                        prop_bits.append(p["select"].get("name",""))
                body = " ".join(b for b in (prop_bits + text_parts) if b).strip()
                out.append(f"{title} : {body[:600]}" if body else title)
            return "\n\n".join(out)
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur read_brain")
        return f"Je n'ai pas pu consulter le brain : {e}"
```
Garder `_extract_notion_title` (deja present). Supprimer l'ancien `search_notion` OU le garder
comme alias. Mettre a jour l'import dans agent.py en consequence.

## Tool 2 — `write_journal` (ecriture append-only dans le Journal)

Schema EXACT de la base Journal (respecter les accents) :
- "Action" : title (resume court de l'action)
- "Detail" -> en Notion c'est "Détail" : rich_text
- "Source" : rich_text (mettre "agent vocal")
- "Agent"  : rich_text (TOUJOURS "Claude (assistant)")
- "Type"   : select, une valeur parmi : "info", "action", "erreur", "décision requise"
- "Créé le": created_time (AUTO, NE PAS ecrire)

```python
@function_tool()
async def write_journal(action: str, detail: str = "", type: str = "info") -> str:
    """Ajoute une entree au Journal du brain Notion de Raphael (trace d'activite).
    A utiliser pour consigner une action faite ou une info importante issue de l'appel.

    Args:
        action: Resume court de ce qui s'est passe (titre).
        detail: Details complementaires (optionnel).
        type: Categorie : 'info', 'action', 'erreur' ou 'décision requise'.
    """
    import os, httpx
    valid = {"info", "action", "erreur", "décision requise"}
    if type not in valid:
        type = "info"
    headers = {
        "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    payload = {
        "parent": {"database_id": "1781b732-9e14-42f7-9c61-ab63e3f8ff0d"},
        "properties": {
            "Action": {"title": [{"text": {"content": action[:200]}}]},
            "Détail": {"rich_text": [{"text": {"content": detail[:1800]}}]},
            "Source": {"rich_text": [{"text": {"content": "agent vocal"}}]},
            "Agent":  {"rich_text": [{"text": {"content": "Claude (assistant)"}}]},
            "Type":   {"select": {"name": type}},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=WRITE_TIMEOUT) as client:
            r = await client.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
            r.raise_for_status()
        return "Note ajoutee au journal."
    except Exception as e:  # noqa: BLE001
        logger.exception("Erreur write_journal")
        return f"Je n'ai pas pu ecrire dans le journal : {e}"
```
ATTENTION : la cle de propriete doit etre EXACTEMENT "Détail" et "Créé le" avec accents.
Ne jamais ecrire "Créé le" (auto). Tester un POST reel et confirmer 200 + page creee.

## agent.py
- Importer `read_brain` et `write_journal` (retirer `search_notion` de l'import s'il est supprime).
- Les ajouter a la liste TOOLS (remplacer search_notion par read_brain).

## system_prompt.py — ajouter
```
BRAIN NOTION :
- Tu peux consulter le cerveau de Raphael avec read_brain (profil, agents, journal, items).
- Tu peux consigner une action ou info importante avec write_journal (type info/action/erreur).
- Ecris dans le journal de facon concise quand une demande a ete traitee pendant l'appel.
- Ne divulgue pas d'informations sensibles du brain a un interlocuteur qui n'est pas Raphael.
```

## Tests apres REBUILD
1. docker compose up -d --build ; verifier "registered worker".
2. Appel : demander "qu'est-ce que tu sais sur mon infra ?" -> read_brain doit renvoyer du contenu.
3. Demander "note dans le journal que j'ai teste l'agent vocal" -> write_journal cree une entree
   (verifier dans Notion : Action remplie, Agent="Claude (assistant)", Type=info, Créé le auto).

## Ne pas toucher
STT nova-3 fr, TTS sonic-2, LLM Anthropic, end_call/inactivite, RoomOutputOptions(audio_enabled=True).
