import json

from src.rag.chunking import _decouper_document, _indexer_documents_longs
from src.rag.local_backend import InMemoryVectorCollection, KeywordEmbeddingModel


def test_decoupe_sans_chevauchement() -> None:
    texte = " ".join(f"mot{i}" for i in range(100))
    chunks = _decouper_document(texte, taille_chunk=30, chevauchement=0)
    assert len(chunks) == 4


def test_decoupe_avec_chevauchement() -> None:
    texte = " ".join(f"mot{i}" for i in range(10))
    chunks = _decouper_document(texte, taille_chunk=5, chevauchement=2)
    assert "mot3" in chunks[0]
    assert "mot3" in chunks[1]
    assert "mot4" in chunks[0]
    assert "mot4" in chunks[1]


def test_chunk_vide() -> None:
    assert _decouper_document("", taille_chunk=30, chevauchement=5) == []


def test_texte_plus_court_que_chunk() -> None:
    texte = " ".join(f"mot{i}" for i in range(50))
    assert _decouper_document(texte, taille_chunk=200, chevauchement=20) == [texte]


def test_metadonnees_source(tmp_path) -> None:
    chemin = tmp_path / "docs.jsonl"
    document = {"id": "doc-1", "titre": "Doc", "categorie": "retours", "texte": "retour " * 80}
    chemin.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")

    collection = InMemoryVectorCollection()
    nb_chunks = _indexer_documents_longs(
        chemin,
        collection,
        KeywordEmbeddingModel(),
        taille_chunk=30,
        chevauchement=5,
    )

    assert nb_chunks > 1
    metadatas = collection.get()["metadatas"]
    assert all(metadata["id_document"] == "doc-1" for metadata in metadatas)

