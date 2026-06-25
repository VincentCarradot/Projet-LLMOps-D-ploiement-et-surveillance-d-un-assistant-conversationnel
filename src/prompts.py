"""Gestion locale du prompt RAG avec fallback Langfuse possible."""

from __future__ import annotations

from pathlib import Path

PROMPT_SYSTEME_DEFAUT = """Tu es un assistant service client.
Reponds uniquement en te basant sur le contexte fourni.
Si la reponse n'est pas dans le contexte, reponds exactement :
"Je n'ai pas l'information."

Contexte :
{{contexte}}

Question : {{question}}

Reponse :"""


def charger_prompt_local(chemin: str | Path = "prompts/default_prompt.txt") -> str:
    """Charge le prompt local, ou retourne le prompt par defaut."""
    path = Path(chemin)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return PROMPT_SYSTEME_DEFAUT


def compiler_prompt(template: str, *, contexte: str, question: str) -> str:
    """Compile un template compatible Langfuse ou str.format."""
    if "{{contexte}}" in template or "{{question}}" in template:
        return template.replace("{{contexte}}", contexte).replace("{{question}}", question)
    return template.format(contexte=contexte, question=question)

