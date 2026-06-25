"""Fonctions simples de tokenisation pour les tests et les metriques."""

from __future__ import annotations

import re

MOTIF_MOT = re.compile(r"\w+", flags=re.UNICODE)


def _tokeniser(texte: str) -> list[str]:
    """Retourne une tokenisation mots simple, suffisante pour ce TP."""
    return MOTIF_MOT.findall(texte.lower())


def _compter_tokens(texte: str) -> int:
    """Compte les tokens selon la tokenisation locale du projet."""
    return len(_tokeniser(texte))


def tronquer_mots(texte: str, nb_mots_max: int) -> str:
    """Tronque un texte au nombre de mots demande."""
    if nb_mots_max <= 0:
        return ""
    mots = texte.split()
    return " ".join(mots[:nb_mots_max])

