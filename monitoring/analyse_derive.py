"""Analyse de derive avec Evidently AI ou fallback statistique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

REFERENCE_DEFAUT = Path("data/corpus_reference.jsonl")
PRODUCTION_DEFAUT = Path("logs/requetes.jsonl")
RAPPORT_HTML_DEFAUT = Path("monitoring/rapport_derive.html")
RAPPORT_JSON_DEFAUT = Path("monitoring/rapport_derive.json")


def _charger_jsonl(chemin: Path) -> pd.DataFrame:
    lignes: list[dict[str, Any]] = []
    if not chemin.exists():
        return pd.DataFrame()
    with chemin.open("r", encoding="utf-8") as fichier:
        for ligne in fichier:
            if ligne.strip():
                lignes.append(json.loads(ligne))
    return pd.DataFrame(lignes)


def _rapport_fallback(reference: pd.DataFrame, courant: pd.DataFrame) -> dict[str, Any]:
    colonnes = ["longueur_prompt", "longueur_reponse", "nb_docs_rag_trouves"]
    drifted = 0
    details = []
    for colonne in colonnes:
        if colonne not in reference or colonne not in courant or courant.empty:
            continue
        moyenne_ref = mean(reference[colonne].dropna().tolist())
        moyenne_cur = mean(courant[colonne].dropna().tolist())
        ratio = abs(moyenne_cur - moyenne_ref) / max(abs(moyenne_ref), 1.0)
        est_drift = ratio > 0.30
        drifted += int(est_drift)
        details.append(
            {
                "column": colonne,
                "reference_mean": moyenne_ref,
                "current_mean": moyenne_cur,
                "relative_change": ratio,
                "drift_detected": est_drift,
            }
        )
    part = drifted / len(details) if details else 0.0
    return {
        "metrics": [
            {
                "metric": "DatasetDriftMetric",
                "result": {
                    "share_of_drifted_columns": part,
                    "details": details,
                    "fallback": True,
                },
            }
        ]
    }


def generer_rapport_derive(
    reference_path: Path = REFERENCE_DEFAUT,
    courant_path: Path = PRODUCTION_DEFAUT,
    sortie_html: Path = RAPPORT_HTML_DEFAUT,
    sortie_json: Path = RAPPORT_JSON_DEFAUT,
) -> dict[str, Any]:
    """Genere un rapport Evidently si disponible, sinon un rapport simple."""
    reference = _charger_jsonl(reference_path)
    courant = _charger_jsonl(courant_path)
    sortie_html.parent.mkdir(parents=True, exist_ok=True)
    sortie_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
        from evidently.report import Report
    except Exception:
        rapport = _rapport_fallback(reference, courant)
        sortie_json.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
        sortie_html.write_text(
            "<html><body><h1>Rapport de derive fallback</h1><pre>"
            + json.dumps(rapport, ensure_ascii=False, indent=2)
            + "</pre></body></html>",
            encoding="utf-8",
        )
        return rapport

    mapping = ColumnMapping(
        numerical_features=["longueur_prompt", "longueur_reponse", "nb_docs_rag_trouves"],
        text_features=["prompt"],
    )
    report = Report(metrics=[DataDriftPreset(), TextOverviewPreset(column_name="prompt")])
    report.run(reference_data=reference, current_data=courant, column_mapping=mapping)
    report.save_html(str(sortie_html))
    rapport = report.as_dict()
    sortie_json.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    return rapport


def detecter_alerte(rapport_json: Path = RAPPORT_JSON_DEFAUT, seuil: float = 0.30) -> bool:
    """Retourne True si la part de colonnes en derive depasse le seuil."""
    if not rapport_json.exists():
        print(f"Rapport absent: {rapport_json}")
        return False
    rapport = json.loads(rapport_json.read_text(encoding="utf-8"))
    for metric in rapport.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            part = metric.get("result", {}).get("share_of_drifted_columns")
            if part is None:
                return False
            alerte = float(part) >= seuil
            message = "ALERTE derive" if alerte else "Pas de derive critique"
            print(f"{message}: share_of_drifted_columns={part:.2f}, seuil={seuil:.2f}")
            return alerte
    print("Metric DatasetDriftMetric introuvable.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse de derive des requetes RAG")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DEFAUT)
    parser.add_argument("--courant", type=Path, default=PRODUCTION_DEFAUT)
    parser.add_argument("--sortie-html", type=Path, default=RAPPORT_HTML_DEFAUT)
    parser.add_argument("--sortie-json", type=Path, default=RAPPORT_JSON_DEFAUT)
    parser.add_argument("--seuil", type=float, default=0.30)
    args = parser.parse_args()
    generer_rapport_derive(args.reference, args.courant, args.sortie_html, args.sortie_json)
    return 1 if detecter_alerte(args.sortie_json, args.seuil) else 0


if __name__ == "__main__":
    raise SystemExit(main())

