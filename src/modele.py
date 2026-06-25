"""Chargement optionnel du LLM et generation avec fallback local."""

from __future__ import annotations

import re
from typing import Any

MODELE_LLM_DEFAUT = "Qwen/Qwen2.5-1.5B-Instruct"
MODELE_EMBEDDING_DEFAUT = "sentence-transformers/all-MiniLM-L6-v2"
MODELE_RERANKING_DEFAUT = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def charger_llm(nom_modele: str = MODELE_LLM_DEFAUT) -> tuple[Any, Any]:
    """Charge un modele HuggingFace causal LM.

    Le chargement est volontairement isole pour que les tests unitaires ne
    telechargent pas de modele lourd.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependances optionnelles
        raise RuntimeError(
            "Installez transformers et torch pour charger le LLM reel."
        ) from exc

    tokeniseur = AutoTokenizer.from_pretrained(nom_modele)
    modele = AutoModelForCausalLM.from_pretrained(
        nom_modele,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return modele, tokeniseur


def _extraire_reponse_depuis_contexte(prompt: str) -> str:
    """Fallback deterministe: extrait une reponse FAQ du contexte injecte."""
    match = re.search(r"Reponse\s*:\s*(.+)", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Je n'ai pas l'information."


def generer_texte(
    prompt: str,
    modele_llm: Any | None = None,
    tokeniseur: Any | None = None,
    nb_tokens_max: int = 200,
) -> str:
    """Genere du texte avec un LLM, un callable de test ou un fallback local."""
    if modele_llm is None:
        return _extraire_reponse_depuis_contexte(prompt)

    if hasattr(modele_llm, "generate_text"):
        return str(modele_llm.generate_text(prompt, max_new_tokens=nb_tokens_max))

    if callable(modele_llm) and tokeniseur is None:
        return str(modele_llm(prompt, max_new_tokens=nb_tokens_max))

    if tokeniseur is None:
        raise ValueError("Un tokeniseur est requis pour utiliser ce modele.")

    entrees = tokeniseur(prompt, return_tensors="pt")
    if hasattr(modele_llm, "device"):
        entrees = {cle: valeur.to(modele_llm.device) for cle, valeur in entrees.items()}
    sorties = modele_llm.generate(
        **entrees,
        max_new_tokens=nb_tokens_max,
        do_sample=False,
        pad_token_id=getattr(tokeniseur, "eos_token_id", None),
    )
    texte = tokeniseur.decode(sorties[0], skip_special_tokens=True)
    if texte.startswith(prompt):
        texte = texte[len(prompt) :]
    return texte.strip()

