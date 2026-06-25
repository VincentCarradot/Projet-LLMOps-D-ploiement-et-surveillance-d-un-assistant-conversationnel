"""Decoupage et indexation de documents longs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag.base_connaissance import _encoder_textes


def _decouper_document(
    texte: str,
    taille_chunk: int = 300,
    chevauchement: int = 50,
) -> list[str]:
    """Decoupe un document en chunks de mots avec chevauchement."""
    if taille_chunk <= 0:
        raise ValueError("taille_chunk doit etre strictement positif.")
    if chevauchement < 0:
        raise ValueError("chevauchement doit etre positif ou nul.")
    if chevauchement >= taille_chunk:
        raise ValueError("chevauchement doit etre inferieur a taille_chunk.")

    mots = texte.split()
    if not mots:
        return []

    pas = taille_chunk - chevauchement
    chunks = []
    for debut in range(0, len(mots), pas):
        fin = min(debut + taille_chunk, len(mots))
        chunks.append(" ".join(mots[debut:fin]))
        if fin == len(mots):
            break
    return chunks


def _indexer_documents_longs(
    chemin_jsonl: str | Path,
    collection: Any,
    modele_embedding: Any,
    taille_chunk: int = 300,
    chevauchement: int = 50,
) -> int:
    """Indexe des documents longs chunkes dans une collection vectorielle."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    with Path(chemin_jsonl).open("r", encoding="utf-8") as fichier:
        for ligne in fichier:
            if not ligne.strip():
                continue
            document_source = json.loads(ligne)
            id_document = str(document_source["id"])
            chunks = _decouper_document(
                str(document_source["texte"]),
                taille_chunk=taille_chunk,
                chevauchement=chevauchement,
            )
            for index, chunk in enumerate(chunks):
                ids.append(f"{id_document}__chunk_{index}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "id_document": id_document,
                        "chunk_index": index,
                        "titre": str(document_source.get("titre", "")),
                        "categorie": str(document_source.get("categorie", "")),
                    }
                )

    if not documents:
        return 0

    embeddings = _encoder_textes(modele_embedding, documents)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)

