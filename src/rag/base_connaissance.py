"""Creation et alimentation de la base de connaissance ChromaDB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.modele import MODELE_EMBEDDING_DEFAUT
from src.rag.local_backend import InMemoryVectorCollection, KeywordEmbeddingModel

CHEMIN_FAQ_DEFAUT = Path("data/faq_service_client.jsonl")
CHEMIN_CHROMA_DEFAUT = Path("data/chroma_db")
NOM_COLLECTION_DEFAUT = "faq_service_client"


def charger_faq(chemin_jsonl: str | Path = CHEMIN_FAQ_DEFAUT) -> list[dict[str, Any]]:
    """Charge une FAQ au format JSON Lines."""
    chemin = Path(chemin_jsonl)
    entrees: list[dict[str, Any]] = []
    with chemin.open("r", encoding="utf-8") as fichier:
        for numero_ligne, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                entree = json.loads(ligne)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalide ligne {numero_ligne}: {chemin}") from exc
            entrees.append(entree)
    return entrees


def formater_entree_faq(entree: dict[str, Any]) -> str:
    """Transforme une entree FAQ en document indexable."""
    return f"Question: {entree['question']}\nReponse: {entree['reponse']}"


def charger_modele_embedding(nom_modele: str = MODELE_EMBEDDING_DEFAUT) -> Any:
    """Charge le modele d'embedding sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependance optionnelle
        raise RuntimeError(
            "Installez sentence-transformers pour utiliser les embeddings reels."
        ) from exc
    return SentenceTransformer(nom_modele)


def _encoder_textes(modele_embedding: Any, textes: list[str]) -> list[list[float]]:
    """Encode une liste de textes en listes Python compatibles ChromaDB."""
    try:
        embeddings = modele_embedding.encode(textes, normalize_embeddings=True)
    except TypeError:
        embeddings = modele_embedding.encode(textes)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vecteur) for vecteur in embeddings]


def _creer_collection_chroma(
    chemin_db: str | Path = CHEMIN_CHROMA_DEFAUT,
    nom_collection: str = NOM_COLLECTION_DEFAUT,
    distance: str = "cosine",
) -> Any:
    """Cree ou ouvre une collection ChromaDB persistante."""
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependance optionnelle
        raise RuntimeError("Installez chromadb pour utiliser la base vectorielle.") from exc

    client = chromadb.PersistentClient(path=str(chemin_db))
    return client.get_or_create_collection(
        name=nom_collection,
        metadata={"hnsw:space": distance},
    )


def _alimenter_collection(
    collection: Any,
    faq: list[dict[str, Any]],
    modele_embedding: Any,
    forcer: bool = False,
) -> Any:
    """Insere les entrees FAQ si la collection est vide."""
    if not forcer and hasattr(collection, "count") and collection.count() > 0:
        return collection

    ids = [str(entree["id"]) for entree in faq]
    documents = [formater_entree_faq(entree) for entree in faq]
    metadatas = [
        {
            "id_source": str(entree["id"]),
            "categorie": str(entree.get("categorie", "")),
            "question": str(entree["question"]),
        }
        for entree in faq
    ]
    embeddings = _encoder_textes(modele_embedding, documents)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return collection


def _creer_base_connaissance(
    chemin_jsonl: str | Path = CHEMIN_FAQ_DEFAUT,
    chemin_db: str | Path = CHEMIN_CHROMA_DEFAUT,
    nom_collection: str = NOM_COLLECTION_DEFAUT,
    modele_embedding: Any | None = None,
) -> Any:
    """Cree la collection ChromaDB persistante et l'alimente."""
    faq = charger_faq(chemin_jsonl)
    modele = modele_embedding or charger_modele_embedding()
    collection = _creer_collection_chroma(chemin_db, nom_collection)
    return _alimenter_collection(collection, faq, modele)


def creer_base_demo(chemin_jsonl: str | Path = CHEMIN_FAQ_DEFAUT) -> tuple[Any, Any]:
    """Cree une base en memoire pour les tests, l'API demo et l'evaluation."""
    faq = charger_faq(chemin_jsonl)
    modele_embedding = KeywordEmbeddingModel()
    collection = InMemoryVectorCollection(name=NOM_COLLECTION_DEFAUT)
    _alimenter_collection(collection, faq, modele_embedding, forcer=True)
    return collection, modele_embedding

