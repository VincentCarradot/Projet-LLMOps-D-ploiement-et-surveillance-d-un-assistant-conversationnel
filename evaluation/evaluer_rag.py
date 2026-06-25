"""Evaluation automatique du pipeline RAG."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RACINE_PROJET = Path(__file__).resolve().parents[1]
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))

from src.rag.base_connaissance import creer_base_demo  # noqa: E402
from src.rag.pipeline import RagConfig, _generer_avec_rag  # noqa: E402
from src.rag.recherche import _rechercher_documents  # noqa: E402

JEU_EVALUATION = Path("evaluation/jeu_evaluation.jsonl")
RAPPORT_SORTIE = Path("evaluation/rapport_eval.json")


def _charger_jsonl(chemin: Path) -> list[dict[str, Any]]:
    lignes: list[dict[str, Any]] = []
    with chemin.open("r", encoding="utf-8") as fichier:
        for ligne in fichier:
            if ligne.strip():
                lignes.append(json.loads(ligne))
    return lignes


def _lcs_longueur(a: list[str], b: list[str]) -> int:
    precedent = [0] * (len(b) + 1)
    for token_a in a:
        courant = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                courant.append(precedent[index_b - 1] + 1)
            else:
                courant.append(max(precedent[index_b], courant[-1]))
        precedent = courant
    return precedent[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    """Calcule un ROUGE-L F1 minimal sans dependance externe."""
    ref = reference.lower().split()
    pred = prediction.lower().split()
    if not ref or not pred:
        return 0.0
    lcs = _lcs_longueur(ref, pred)
    precision = lcs / len(pred)
    rappel = lcs / len(ref)
    if precision + rappel == 0:
        return 0.0
    return 2 * precision * rappel / (precision + rappel)


def evaluer_retrieval(
    jeu: list[dict[str, Any]],
    collection: Any,
    modele_embedding: Any,
    k: int,
) -> tuple[float, float]:
    """Retourne Recall@k et MRR pour les questions avec docs pertinents."""
    recalls = []
    reciprocal_ranks = []
    for entree in jeu:
        pertinents = set(entree.get("docs_pertinents", []))
        if not pertinents:
            continue
        resultats = _rechercher_documents(
            entree["question"],
            collection,
            modele_embedding,
            top_k=k,
            seuil_similarite=0.0,
        )
        ids = [resultat["id"] for resultat in resultats]
        recalls.append(float(bool(pertinents.intersection(ids))))
        rang = 0.0
        for index, doc_id in enumerate(ids, start=1):
            if doc_id in pertinents:
                rang = 1 / index
                break
        reciprocal_ranks.append(rang)
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return recall, mrr


def evaluer_generation(
    jeu: list[dict[str, Any]],
    collection: Any,
    modele_embedding: Any,
) -> float:
    """Calcule le ROUGE-L moyen des reponses RAG."""
    scores = []
    for entree in jeu:
        resultat = _generer_avec_rag(
            entree["question"],
            collection,
            modele_embedding,
            config=RagConfig(top_k_biencoder=5, top_k_final=3, seuil_similarite=0.2),
        )
        scores.append(rouge_l_f1(entree["reponse_reference"], resultat["reponse"]))
    return sum(scores) / len(scores) if scores else 0.0


def construire_rapport() -> dict[str, Any]:
    jeu = _charger_jsonl(JEU_EVALUATION)
    collection, modele_embedding = creer_base_demo()
    recall_1, _ = evaluer_retrieval(jeu, collection, modele_embedding, k=1)
    recall_3, mrr = evaluer_retrieval(jeu, collection, modele_embedding, k=3)
    recall_5, _ = evaluer_retrieval(jeu, collection, modele_embedding, k=5)
    rouge_l = evaluer_generation(jeu, collection, modele_embedding)
    rapport = {
        "recall_at_1": round(recall_1, 4),
        "recall_at_3": round(recall_3, 4),
        "recall_at_5": round(recall_5, 4),
        "mrr": round(mrr, 4),
        "rouge_l_moyen": round(rouge_l, 4),
        "nb_questions": len(jeu),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return rapport


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation du RAG")
    parser.add_argument("--sortie", type=Path, default=RAPPORT_SORTIE)
    parser.add_argument("--seuil-recall3", type=float, default=0.75)
    parser.add_argument("--seuil-rouge-l", type=float, default=0.20)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    rapport = construire_rapport()
    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    args.sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rapport, ensure_ascii=False, indent=2))

    alertes = []
    if rapport["recall_at_3"] < args.seuil_recall3:
        alertes.append(f"Recall@3={rapport['recall_at_3']} < {args.seuil_recall3}")
    if rapport["rouge_l_moyen"] < args.seuil_rouge_l:
        alertes.append(f"ROUGE-L={rapport['rouge_l_moyen']} < {args.seuil_rouge_l}")
    if alertes:
        print("Alertes evaluation: " + "; ".join(alertes), file=sys.stderr)
        return 1 if args.fail_on_threshold else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
