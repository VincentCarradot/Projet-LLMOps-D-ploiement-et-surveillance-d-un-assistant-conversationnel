"""API FastAPI exposant le pipeline RAG."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependance optionnelle
    def load_dotenv() -> bool:
        return False

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring import prometheus_app as metrics
from src.modele import MODELE_LLM_DEFAUT, charger_llm
from src.prompts import charger_prompt_local
from src.rag.base_connaissance import (
    _creer_base_connaissance,
    charger_modele_embedding,
    creer_base_demo,
)
from src.rag.pipeline import RagConfig, _generer_avec_rag
from src.rag.reranking import charger_crossencoder

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover - dependance optionnelle
    Instrumentator = None

LOGGER = logging.getLogger("llmops.api")
LOG_PATH = Path("logs/requetes.jsonl")

app = FastAPI(
    title="Assistant service client RAG",
    version="0.1.0",
    description="API locale pour le TP LLMOps: RAG, traces, evaluation et monitoring.",
)

if Instrumentator is not None:  # pragma: no branch
    Instrumentator().instrument(app).expose(app)


class _LangfuseObservationAdapter:
    """Adapte l'API Langfuse v4 a l'interface interne du pipeline."""

    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def end(self, output: Any | None = None, usage: dict[str, int] | None = None) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if usage is not None:
            kwargs["usage_details"] = usage
        if kwargs and hasattr(self._observation, "update"):
            self._observation.update(**kwargs)
        if hasattr(self._observation, "end"):
            self._observation.end()


class _LangfuseTraceAdapter:
    """Trace racine compatible avec les methodes attendues par le pipeline."""

    def __init__(
        self,
        client: Any,
        *,
        name: str,
        input: Any,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self._client = client
        self.trace_id = client.create_trace_id()
        metadata = dict(metadata)
        if user_id:
            metadata["user_id"] = user_id
        self._root = client.start_observation(
            trace_context={"trace_id": self.trace_id},
            name=name,
            as_type="span",
            input=input,
            metadata=metadata,
        )
        if hasattr(self._root, "set_trace_io"):
            self._root.set_trace_io(input=input)

    def span(
        self,
        *,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _LangfuseObservationAdapter:
        observation = self._root.start_observation(
            name=name,
            as_type="span",
            input=input,
            metadata=metadata,
        )
        return _LangfuseObservationAdapter(observation)

    def generation(
        self,
        *,
        name: str,
        model: str,
        input: Any | None = None,
    ) -> _LangfuseObservationAdapter:
        observation = self._root.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=input,
        )
        return _LangfuseObservationAdapter(observation)

    def score(self, *, name: str, value: float | int | str) -> None:
        if hasattr(self._root, "score_trace"):
            self._root.score_trace(name=name, value=value)
        else:
            self._client.create_score(trace_id=self.trace_id, name=name, value=value)

    def update(self, *, output: Any | None = None) -> None:
        if hasattr(self._root, "set_trace_io"):
            self._root.set_trace_io(output=output)
        if hasattr(self._root, "update"):
            self._root.update(output=output)
        if hasattr(self._root, "end"):
            self._root.end()


class GenerateRequest(BaseModel):
    """Schema de requete de generation."""

    prompt: str = Field(min_length=1, max_length=2000)
    nb_tokens_max: int = Field(default=200, ge=1, le=500)
    top_k_final: int = Field(default=3, ge=1, le=5)
    user_id: str | None = Field(default=None, max_length=120)


class GenerateResponse(BaseModel):
    """Schema de reponse de generation."""

    reponse: str
    nb_tokens: int
    sources: list[str]
    hors_domaine: bool
    duree_ms: float
    trace_url: str | None = None


def _timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initialiser_langfuse() -> Any | None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse()
        if hasattr(client, "auth_check"):
            client.auth_check()
        return client
    except Exception as exc:  # pragma: no cover - service externe
        LOGGER.warning("Langfuse indisponible: %s", exc)
        return None


def _charger_prompt(langfuse: Any | None) -> str:
    if langfuse is not None:
        try:
            prompt = langfuse.get_prompt("assistant-service-client")
            contenu = getattr(prompt, "prompt", None)
            if isinstance(contenu, str):
                return contenu
        except Exception as exc:  # pragma: no cover - service externe
            LOGGER.warning("Prompt Langfuse indisponible, fallback local: %s", exc)
    return charger_prompt_local()


def _creer_trace_langfuse(
    langfuse: Any,
    *,
    name: str,
    input: Any,
    user_id: str | None,
    metadata: dict[str, Any],
) -> Any:
    """Cree une trace avec l'ancien SDK Langfuse ou avec le SDK v4."""
    if hasattr(langfuse, "trace"):
        return langfuse.trace(
            name=name,
            input=input,
            user_id=user_id,
            metadata=metadata,
        )
    return _LangfuseTraceAdapter(
        langfuse,
        name=name,
        input=input,
        user_id=user_id,
        metadata=metadata,
    )


def _journaliser_requete(prompt: str, resultat: dict[str, Any], duree_ms: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entree = {
        "timestamp": _timestamp_iso(),
        "prompt": prompt,
        "reponse": resultat["reponse"],
        "longueur_prompt": len(prompt.split()),
        "longueur_reponse": len(resultat["reponse"].split()),
        "duree_ms": round(duree_ms, 2),
        "ids_sources": resultat["sources"],
        "nb_docs_rag_trouves": len(resultat["sources"]),
        "score_reranking_max": resultat["score_reranking_max"],
        "hors_domaine": int(resultat["hors_domaine"]),
    }
    with LOG_PATH.open("a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(entree, ensure_ascii=False) + "\n")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=422,
        content={"erreur": "validation", "details": exc.errors()},
    )


@app.on_event("startup")
async def startup() -> None:
    """Charge les ressources une seule fois au demarrage."""
    load_dotenv(".env")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    app.state.langfuse = _initialiser_langfuse()
    app.state.langfuse_enabled = app.state.langfuse is not None
    app.state.prompt_template = _charger_prompt(app.state.langfuse)
    app.state.mode = "demo"
    app.state.modele_llm = None
    app.state.tokeniseur = None
    app.state.modele_crossencoder = None

    utiliser_modeles_reels = os.getenv("LLMOPS_REAL_MODE", "0") == "1"
    try:
        if utiliser_modeles_reels:
            modele_embedding = charger_modele_embedding()
            collection = _creer_base_connaissance(modele_embedding=modele_embedding)
            if os.getenv("LLMOPS_LOAD_LLM", "0") == "1":
                app.state.modele_llm, app.state.tokeniseur = charger_llm(MODELE_LLM_DEFAUT)
            if os.getenv("LLMOPS_LOAD_RERANKER", "0") == "1":
                app.state.modele_crossencoder = charger_crossencoder()
            app.state.mode = "real"
        else:
            collection, modele_embedding = creer_base_demo()
    except Exception as exc:
        LOGGER.warning("Mode reel indisponible, bascule en demo: %s", exc)
        collection, modele_embedding = creer_base_demo()
        app.state.mode = "demo"

    app.state.collection = collection
    app.state.modele_embedding = modele_embedding


@app.get("/health")
async def health() -> dict[str, Any]:
    collection = getattr(app.state, "collection", None)
    return {
        "status": "ok" if collection is not None else "degraded",
        "mode": getattr(app.state, "mode", "unknown"),
        "langfuse_enabled": bool(getattr(app.state, "langfuse_enabled", False)),
        "modele": MODELE_LLM_DEFAUT,
        "documents": collection.count() if collection is not None and hasattr(collection, "count") else None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    collection = getattr(app.state, "collection", None)
    modele_embedding = getattr(app.state, "modele_embedding", None)
    if collection is None or modele_embedding is None:
        raise HTTPException(status_code=503, detail="Pipeline RAG non initialise.")

    debut = perf_counter()
    trace = None
    trace_url = None
    langfuse = getattr(app.state, "langfuse", None)
    if langfuse is not None:
        trace = _creer_trace_langfuse(
            langfuse,
            name="rag-generate",
            input=payload.prompt,
            user_id=payload.user_id,
            metadata={
                "mode": getattr(app.state, "mode", "unknown"),
                "nb_tokens_max": payload.nb_tokens_max,
                "timestamp": _timestamp_iso(),
            },
        )

    resultat = _generer_avec_rag(
        payload.prompt,
        collection,
        modele_embedding,
        modele_llm=getattr(app.state, "modele_llm", None),
        tokeniseur=getattr(app.state, "tokeniseur", None),
        modele_crossencoder=getattr(app.state, "modele_crossencoder", None),
        config=RagConfig(nb_tokens_max=payload.nb_tokens_max, top_k_final=payload.top_k_final),
        trace=trace,
        prompt_template=getattr(app.state, "prompt_template", charger_prompt_local()),
    )
    duree_ms = (perf_counter() - debut) * 1000

    metrics.rag_retrieval_duration_seconds.observe(resultat["duree_retrieval_ms"] / 1000)
    metrics.rag_reranking_duration_seconds.observe(resultat["duree_reranking_ms"] / 1000)
    metrics.llm_generation_duration_seconds.observe(resultat["duree_generation_ms"] / 1000)
    metrics.rag_docs_retrieved_per_request.observe(len(resultat["sources"]))
    metrics.rag_docs_retrieved_total.inc(len(resultat["sources"]))
    metrics.llm_tokens_generated_total.inc(resultat["nb_tokens"])

    if trace is not None:
        trace.score(name="nb_docs_retrouves", value=len(resultat["sources"]))
        trace.score(name="score_reranking_max", value=resultat["score_reranking_max"])
        trace.score(name="longueur_reponse_mots", value=len(resultat["reponse"].split()))
        trace.score(name="hors_domaine", value=int(resultat["hors_domaine"]))
        trace.update(output=resultat["reponse"])
        if hasattr(langfuse, "flush"):
            langfuse.flush()
        trace_id = getattr(trace, "trace_id", None) or getattr(trace, "id", None)
        if hasattr(langfuse, "get_trace_url") and trace_id is not None:
            trace_url = langfuse.get_trace_url(trace_id=trace_id)

    _journaliser_requete(payload.prompt, resultat, duree_ms)

    return GenerateResponse(
        reponse=resultat["reponse"],
        nb_tokens=resultat["nb_tokens"],
        sources=resultat["sources"],
        hors_domaine=resultat["hors_domaine"],
        duree_ms=round(duree_ms, 2),
        trace_url=trace_url,
    )
