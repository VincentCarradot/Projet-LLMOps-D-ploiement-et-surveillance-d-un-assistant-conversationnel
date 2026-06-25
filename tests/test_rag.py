from src.rag.pipeline import RagConfig, _construire_prompt_augmente, _generer_avec_rag
from src.rag.recherche import _rechercher_documents
from src.rag.reranking import _reclasser_passages
from tests.dataset import FakeCrossEncoder, FakeLLM, collection_demo


def test_recherche_retourne_resultats() -> None:
    collection, modele_embedding = collection_demo()
    resultats = _rechercher_documents(
        "retourner un article",
        collection,
        modele_embedding,
        top_k=3,
        seuil_similarite=0.0,
    )
    assert len(resultats) == 3


def test_recherche_pertinence() -> None:
    collection, modele_embedding = collection_demo()
    resultats = _rechercher_documents(
        "retourner un article",
        collection,
        modele_embedding,
        top_k=5,
        seuil_similarite=0.1,
    )
    assert "faq-01" in {resultat["id"] for resultat in resultats}


def test_recherche_hors_domaine() -> None:
    collection, modele_embedding = collection_demo()
    resultats = _rechercher_documents(
        "quelle météo demain à Tokyo",
        collection,
        modele_embedding,
        top_k=3,
        seuil_similarite=0.35,
    )
    assert resultats == []


def test_prompt_augmente_contient_contexte() -> None:
    documents = [{"id": "faq-01", "texte": "Les retours sont acceptés sous 30 jours."}]
    prompt = _construire_prompt_augmente("Quel est le délai de retour ?", documents)
    assert "Les retours sont acceptés sous 30 jours." in prompt
    assert "Quel est le délai de retour ?" in prompt


def test_pipeline_complet_retourne_str() -> None:
    collection, modele_embedding = collection_demo()
    resultat = _generer_avec_rag(
        "Sous quel délai puis-je retourner un article ?",
        collection,
        modele_embedding,
        modele_llm=FakeLLM(),
        config=RagConfig(top_k_biencoder=5, top_k_final=3, seuil_similarite=0.1),
    )
    assert isinstance(resultat["reponse"], str)
    assert resultat["reponse"]
    assert "faq-01" in resultat["sources"]


def test_reranking_retourne_top_k() -> None:
    candidats = [
        {"id": "a", "texte": "Reponse: lien de suivi envoyé"},
        {"id": "b", "texte": "Reponse: retour possible sous 30 jours"},
        {"id": "c", "texte": "Reponse: carte cadeau disponible"},
    ]
    resultats = _reclasser_passages("je veux retourner un article", candidats, FakeCrossEncoder(), 2)
    assert len(resultats) == 2


def test_reranking_ordre_coherent() -> None:
    candidats = [
        {"id": "a", "texte": "Reponse: carte cadeau disponible"},
        {"id": "b", "texte": "Reponse: retour possible sous 30 jours"},
    ]
    resultats = _reclasser_passages("retourner un article", candidats, FakeCrossEncoder(), 2)
    assert resultats[0]["id"] == "b"


def test_reranking_integre_pipeline() -> None:
    collection, modele_embedding = collection_demo()
    resultat = _generer_avec_rag(
        "je veux retourner un article",
        collection,
        modele_embedding,
        modele_llm=FakeLLM(),
        modele_crossencoder=FakeCrossEncoder(),
        config=RagConfig(top_k_biencoder=5, top_k_final=2, seuil_similarite=0.1),
    )
    assert resultat["reponse"]
    assert resultat["score_reranking_max"] > 0

