"""Reranking des passages candidats avec un cross-encoder."""

from __future__ import annotations

import math
from typing import Any

from src.modele import MODELE_RERANKING_DEFAUT


def charger_crossencoder(nom_modele: str = MODELE_RERANKING_DEFAUT) -> Any:
    """Charge le cross-encoder sentence-transformers."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependance optionnelle
        raise RuntimeError(
            "Installez sentence-transformers pour utiliser le reranking reel."
        ) from exc
    return CrossEncoder(nom_modele)


def _sigmoide(valeur: float) -> float:
    return 1.0 / (1.0 + math.exp(-valeur))


def _texte_candidat(candidat: str | dict[str, Any]) -> str:
    if isinstance(candidat, str):
        return candidat
    return str(candidat.get("texte", ""))


def _reclasser_passages(
    requete: str,
    candidats: list[str | dict[str, Any]],
    modele_crossencoder: Any,
    top_k_final: int = 3,
) -> list[dict[str, Any]]:
    """Reclasse les candidats par pertinence decroissante."""
    if top_k_final <= 0 or not candidats:
        return []

    paires = [(requete, _texte_candidat(candidat)) for candidat in candidats]
    scores_bruts = modele_crossencoder.predict(paires)
    if hasattr(scores_bruts, "tolist"):
        scores_bruts = scores_bruts.tolist()

    resultats: list[dict[str, Any]] = []
    for candidat, score_brut in zip(candidats, scores_bruts, strict=True):
        entree = dict(candidat) if isinstance(candidat, dict) else {"texte": candidat}
        brut = float(score_brut)
        entree["score_reranking_brut"] = brut
        entree["score_reranking"] = _sigmoide(brut)
        resultats.append(entree)

    resultats.sort(key=lambda item: item["score_reranking_brut"], reverse=True)
    return resultats[:top_k_final]

