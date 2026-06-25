"""Metriques Prometheus personnalisees pour l'API RAG."""

from __future__ import annotations


class _MetricNulle:
    def observe(self, valeur: float) -> None:
        del valeur

    def inc(self, valeur: float = 1.0) -> None:
        del valeur


try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover - dependance optionnelle
    rag_retrieval_duration_seconds = _MetricNulle()
    rag_reranking_duration_seconds = _MetricNulle()
    llm_generation_duration_seconds = _MetricNulle()
    rag_docs_retrieved_per_request = _MetricNulle()
    rag_docs_retrieved_total = _MetricNulle()
    llm_tokens_generated_total = _MetricNulle()
else:
    rag_retrieval_duration_seconds = Histogram(
        "rag_retrieval_duration_seconds",
        "Duree de la recherche vectorielle RAG.",
    )
    rag_reranking_duration_seconds = Histogram(
        "rag_reranking_duration_seconds",
        "Duree du reranking cross-encoder.",
    )
    llm_generation_duration_seconds = Histogram(
        "llm_generation_duration_seconds",
        "Duree de generation LLM.",
    )
    rag_docs_retrieved_per_request = Histogram(
        "rag_docs_retrieved_per_request",
        "Nombre de documents RAG injectes par requete.",
        buckets=(0, 1, 2, 3, 5, 10),
    )
    rag_docs_retrieved_total = Counter(
        "rag_docs_retrieved_total",
        "Nombre total de documents RAG retrouves.",
    )
    llm_tokens_generated_total = Counter(
        "llm_tokens_generated_total",
        "Nombre total de tokens generes par le LLM.",
    )

