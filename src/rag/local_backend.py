"""Backend vectoriel leger utilise pour les tests et le mode demo.

Le TP demande ChromaDB et sentence-transformers pour la production locale.
Ce module evite de rendre les tests dependants de telechargements de modeles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _normaliser(vecteur: list[float]) -> list[float]:
    norme = math.sqrt(sum(valeur * valeur for valeur in vecteur))
    if norme == 0:
        return vecteur
    return [valeur / norme for valeur in vecteur]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    produit = sum(x * y for x, y in zip(a, b, strict=False))
    return 1.0 - produit


class KeywordEmbeddingModel:
    """Embedding symbolique par familles de mots, stable et tres rapide."""

    groupes = (
        ("retour", "retourner", "renvoyer", "article"),
        ("rembours", "remboursement", "argent"),
        ("suivi", "expedition", "expedie", "commande"),
        ("livraison", "livrer", "international", "adresse"),
        ("colis", "endommage", "abime", "photo", "litige"),
        ("annuler", "annulation", "validation"),
        ("cadeau", "carte", "bon"),
        ("compte", "mot", "passe", "connexion"),
        ("garantie", "panne", "defaut"),
        ("facture", "tva", "justificatif"),
        ("produit", "stock", "taille"),
        ("support", "contact", "client"),
    )

    def encode(self, textes: list[str] | str, normalize_embeddings: bool = True) -> list[list[float]]:
        if isinstance(textes, str):
            textes = [textes]
        vecteurs: list[list[float]] = []
        for texte in textes:
            texte_min = texte.lower()
            vecteur = [
                float(sum(1 for mot in groupe if mot in texte_min))
                for groupe in self.groupes
            ]
            vecteurs.append(_normaliser(vecteur) if normalize_embeddings else vecteur)
        return vecteurs


@dataclass
class InMemoryVectorCollection:
    """Subset de l'API ChromaDB suffisant pour les tests du TP."""

    name: str = "collection_demo"
    _documents: dict[str, str] = field(default_factory=dict)
    _metadatas: dict[str, dict[str, Any]] = field(default_factory=dict)
    _embeddings: dict[str, list[float]] = field(default_factory=dict)

    def count(self) -> int:
        return len(self._documents)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        embeddings = embeddings or [[0.0] for _ in documents]
        metadatas = metadatas or [{} for _ in documents]
        for doc_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas, strict=True
        ):
            self._documents[doc_id] = document
            self._embeddings[doc_id] = list(embedding)
            self._metadatas[doc_id] = dict(metadata)

    def get(self, ids: list[str] | None = None) -> dict[str, list[Any]]:
        ids_retour = ids if ids is not None else list(self._documents)
        ids_existants = [doc_id for doc_id in ids_retour if doc_id in self._documents]
        return {
            "ids": ids_existants,
            "documents": [self._documents[doc_id] for doc_id in ids_existants],
            "metadatas": [self._metadatas[doc_id] for doc_id in ids_existants],
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 3,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        requete = query_embeddings[0]
        lignes = []
        for doc_id, embedding in self._embeddings.items():
            distance = _cosine_distance(requete, embedding)
            lignes.append((distance, doc_id))
        lignes.sort(key=lambda item: item[0])
        lignes = lignes[:n_results]
        ids = [doc_id for _, doc_id in lignes]
        distances = [distance for distance, _ in lignes]
        return {
            "ids": [ids],
            "documents": [[self._documents[doc_id] for doc_id in ids]],
            "metadatas": [[self._metadatas[doc_id] for doc_id in ids]],
            "distances": [distances],
        }

