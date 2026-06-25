"""Fixtures et doubles de test pour eviter les modeles lourds."""

from __future__ import annotations

from typing import Any

from src.rag.base_connaissance import creer_base_demo


class FakeCrossEncoder:
    """Cross-encoder deterministe base sur des mots-clés."""

    def predict(self, paires: list[tuple[str, str]]) -> list[float]:
        scores = []
        for requete, document in paires:
            texte = f"{requete} {document}".lower()
            score = 0.0
            if any(mot in texte for mot in ("retour", "retourner", "renvoyer")):
                score += 3.0 if "30 jours" in texte else 1.0
            if "suivi" in texte:
                score += 2.0
            if "meteo" in texte or "météo" in texte:
                score -= 3.0
            scores.append(score)
        return scores


class FakeLLM:
    """LLM de test qui repond a partir du contexte."""

    def generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        del max_new_tokens
        for ligne in prompt.splitlines():
            if ligne.lower().startswith("reponse:"):
                return ligne.split(":", 1)[1].strip()
        return "Je n'ai pas l'information."


def collection_demo() -> tuple[Any, Any]:
    return creer_base_demo()

