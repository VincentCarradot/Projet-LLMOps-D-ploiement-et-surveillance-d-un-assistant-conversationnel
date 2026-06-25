"""Recherche semantique dans une collection vectorielle."""

from __future__ import annotations

from typing import Any

from src.rag.base_connaissance import _encoder_textes


def _similarite_depuis_distance(distance: float) -> float:
    """Convertit une distance cosinus ChromaDB en score de similarite."""
    return max(-1.0, min(1.0, 1.0 - float(distance)))


def _premiere_liste(resultat: dict[str, Any], cle: str) -> list[Any]:
    valeur = resultat.get(cle, [[]])
    if not valeur:
        return []
    return list(valeur[0])


def _rechercher_documents(
    requete: str,
    collection: Any,
    modele_embedding: Any,
    top_k: int = 3,
    seuil_similarite: float = 0.35,
) -> list[dict[str, Any]]:
    """Recherche les documents les plus proches de la requete utilisateur."""
    if top_k <= 0:
        return []

    embedding_requete = _encoder_textes(modele_embedding, [requete])[0]
    try:
        resultat = collection.query(
            query_embeddings=[embedding_requete],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except TypeError:
        resultat = collection.query(query_embeddings=[embedding_requete], n_results=top_k)

    ids = _premiere_liste(resultat, "ids")
    documents = _premiere_liste(resultat, "documents")
    metadatas = _premiere_liste(resultat, "metadatas")
    distances = _premiere_liste(resultat, "distances")

    trouves: list[dict[str, Any]] = []
    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        score = _similarite_depuis_distance(float(distance))
        if score < seuil_similarite:
            continue
        metadata = metadata or {}
        trouves.append(
            {
                "id": str(metadata.get("id_source") or metadata.get("id_document") or doc_id),
                "id_chroma": str(doc_id),
                "texte": str(document),
                "metadata": metadata,
                "distance": float(distance),
                "score": score,
            }
        )
    return trouves

