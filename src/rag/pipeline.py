"""Assemblage du pipeline RAG complet."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.modele import generer_texte
from src.prompts import PROMPT_SYSTEME_DEFAUT, compiler_prompt
from src.rag.recherche import _rechercher_documents
from src.rag.reranking import _reclasser_passages
from src.tokeniseur import _compter_tokens


@dataclass(frozen=True)
class RagConfig:
    """Parametres principaux du pipeline RAG."""

    top_k_biencoder: int = 10
    top_k_final: int = 3
    seuil_similarite: float = 0.35
    nb_tokens_max: int = 200


class _SpanNul:
    def end(self, output: Any | None = None) -> None:
        del output


def _ouvrir_span(trace: Any | None, nom: str, entree: Any, metadata: dict[str, Any]) -> Any:
    if trace is None or not hasattr(trace, "span"):
        return _SpanNul()
    return trace.span(name=nom, input=entree, metadata=metadata)


def _construire_prompt_augmente(
    question: str,
    documents: list[dict[str, Any]],
    template: str = PROMPT_SYSTEME_DEFAUT,
) -> str:
    """Construit le prompt avec les passages sources injectes."""
    contexte = "\n\n".join(
        f"[{index}] source={document['id']}\n{document['texte']}"
        for index, document in enumerate(documents, start=1)
    )
    return compiler_prompt(template, contexte=contexte, question=question)


def _generer_avec_rag(
    prompt_utilisateur: str,
    collection: Any,
    modele_embedding: Any,
    modele_llm: Any | None = None,
    tokeniseur: Any | None = None,
    modele_crossencoder: Any | None = None,
    config: RagConfig | None = None,
    trace: Any | None = None,
    prompt_template: str = PROMPT_SYSTEME_DEFAUT,
) -> dict[str, Any]:
    """Execute retrieval, reranking optionnel, generation et tracabilite."""
    config = config or RagConfig()

    debut_retrieval = perf_counter()
    span_retrieval = _ouvrir_span(
        trace,
        "retrieval",
        prompt_utilisateur,
        {"top_k": config.top_k_biencoder, "seuil": config.seuil_similarite},
    )
    candidats = _rechercher_documents(
        prompt_utilisateur,
        collection,
        modele_embedding,
        top_k=config.top_k_biencoder,
        seuil_similarite=config.seuil_similarite,
    )
    duree_retrieval_ms = (perf_counter() - debut_retrieval) * 1000
    span_retrieval.end(
        output=[{"id": item["id"], "score": item["score"]} for item in candidats]
    )

    debut_reranking = perf_counter()
    span_reranking = _ouvrir_span(
        trace,
        "reranking",
        [{"id": item["id"], "score": item["score"]} for item in candidats],
        {"top_k_final": config.top_k_final, "enabled": modele_crossencoder is not None},
    )
    if modele_crossencoder is not None and candidats:
        documents = _reclasser_passages(
            prompt_utilisateur,
            candidats,
            modele_crossencoder,
            top_k_final=config.top_k_final,
        )
    else:
        documents = candidats[: config.top_k_final]
        for document in documents:
            document.setdefault("score_reranking", document.get("score", 0.0))
    duree_reranking_ms = (perf_counter() - debut_reranking) * 1000
    span_reranking.end(
        output=[
            {"id": item["id"], "score_reranking": item["score_reranking"]}
            for item in documents
        ]
    )

    if not documents:
        reponse = "Je n'ai pas l'information."
        return {
            "reponse": reponse,
            "sources": [],
            "documents": [],
            "prompt_augmente": "",
            "nb_tokens": _compter_tokens(reponse),
            "hors_domaine": True,
            "score_reranking_max": 0.0,
            "duree_retrieval_ms": duree_retrieval_ms,
            "duree_reranking_ms": duree_reranking_ms,
            "duree_generation_ms": 0.0,
        }

    prompt_augmente = _construire_prompt_augmente(prompt_utilisateur, documents, prompt_template)

    debut_generation = perf_counter()
    if trace is not None and hasattr(trace, "generation"):
        generation = trace.generation(
            name="llm",
            model=getattr(modele_llm, "name_or_path", "local-fallback"),
            input=prompt_augmente,
        )
    else:
        generation = None

    reponse = generer_texte(
        prompt_augmente,
        modele_llm=modele_llm,
        tokeniseur=tokeniseur,
        nb_tokens_max=config.nb_tokens_max,
    )
    duree_generation_ms = (perf_counter() - debut_generation) * 1000
    if generation is not None:
        generation.end(
            output=reponse,
            usage={
                "input_tokens": _compter_tokens(prompt_augmente),
                "output_tokens": _compter_tokens(reponse),
                "total_tokens": _compter_tokens(prompt_augmente) + _compter_tokens(reponse),
            },
        )

    return {
        "reponse": reponse,
        "sources": [document["id"] for document in documents],
        "documents": documents,
        "prompt_augmente": prompt_augmente,
        "nb_tokens": _compter_tokens(reponse),
        "hors_domaine": False,
        "score_reranking_max": max(
            float(document.get("score_reranking", 0.0)) for document in documents
        ),
        "duree_retrieval_ms": duree_retrieval_ms,
        "duree_reranking_ms": duree_reranking_ms,
        "duree_generation_ms": duree_generation_ms,
    }
