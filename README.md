# Assistant Service Client RAG - Projet LLMOps

Projet complet de mise en production d'un assistant conversationnel de service client base sur un pipeline **RAG** (*Retrieval-Augmented Generation*), avec API, observabilite LLM, evaluation automatique, detection de derive, monitoring systeme et integration continue.

Ce projet a ete realise dans le cadre d'un TP/projet LLMOps. Il ne se limite pas a faire repondre un modele: il montre comment construire, tester, exposer, surveiller et evaluer un systeme LLM de maniere professionnelle.

## Objectifs

- Repondre aux questions client a partir d'une base de connaissance controlee.
- Reduire les hallucinations grace au RAG et a un prompt contraint.
- Tracer chaque etape du pipeline avec Langfuse.
- Mesurer la qualite du retrieval et de la generation.
- Detecter les derives de donnees en production.
- Exposer le systeme via une API REST FastAPI.
- Automatiser la qualite de code avec pre-commit et GitHub Actions.

## Fonctionnalites principales

| Domaine | Fonctionnalite |
| --- | --- |
| RAG | Recherche semantique, prompt augmente, sources restituees |
| Base vectorielle | ChromaDB en mode persistant, backend memoire pour tests rapides |
| Embeddings | `sentence-transformers` avec `all-MiniLM-L6-v2` en mode reel |
| Reranking | Cross-encoder pour reclasser les passages candidats |
| API | FastAPI avec endpoints `/health`, `/generate`, `/metrics` |
| Observabilite LLM | Traces Langfuse avec spans `retrieval`, `reranking`, `llm` |
| Evaluation | Recall@k, MRR, ROUGE-L, rapport JSON |
| Monitoring | Prometheus, Grafana, metriques custom RAG/LLM |
| Derive | Analyse Evidently AI sur les logs de production |
| Qualite | Ruff, mypy, pytest, coverage, pre-commit |
| CI/CD | Workflows GitHub Actions pour tests, qualite et evaluation RAG |

## Architecture

```text
Client
  |
  | POST /generate
  v
FastAPI
  |
  | 1. Embedding de la question
  v
Recherche semantique
  |
  | 2. Top-k candidats
  v
ChromaDB / backend vectoriel
  |
  | 3. Reranking optionnel
  v
Cross-encoder
  |
  | 4. Prompt augmente
  v
LLM local / fallback demo
  |
  | 5. Reponse + sources
  v
Client
```

Les appels sont instrumentes avec:

- **Langfuse** pour les traces LLM et les scores de qualite.
- **Prometheus** pour les metriques systeme et applicatives.
- **Grafana** pour la visualisation.
- **Evidently AI** pour la detection de derive.

## Stack technique

| Technologie | Utilisation |
| --- | --- |
| Python | Langage principal du projet |
| FastAPI | API REST de generation |
| Pydantic | Validation des schemas d'entree/sortie |
| ChromaDB | Base vectorielle locale persistante |
| sentence-transformers | Embeddings et reranking |
| Transformers / Torch | Chargement du LLM local |
| Qwen2.5-1.5B-Instruct | Modele de generation recommande |
| Langfuse | Observabilite LLM, traces, spans, scores |
| Evidently AI | Analyse de derive des donnees |
| Prometheus | Collecte de metriques |
| Grafana | Dashboard de supervision |
| pytest | Tests unitaires |
| coverage | Couverture de code |
| Ruff | Linting et formatage Python |
| mypy | Verification statique des types |
| pre-commit | Hooks qualite avant commit |
| GitHub Actions | Integration continue |
| Docker Compose | Monitoring local Prometheus/Grafana |

## Structure du depot

```text
.
├── api/
│   └── app.py                      # API FastAPI instrumentee
├── data/
│   ├── faq_service_client.jsonl    # Base FAQ principale
│   ├── documents_longs.jsonl       # Documents longs pour chunking
│   └── corpus_reference.jsonl      # Donnees de reference pour derive
├── docs/
│   ├── guide_pas_a_pas.md          # Guide pedagogique
│   ├── rapport.md                  # Rapport technique detaille
│   └── soutenance.md               # Plan de presentation orale
├── evaluation/
│   ├── jeu_evaluation.jsonl        # Questions de test RAG
│   ├── evaluer_rag.py              # Evaluation Recall/MRR/ROUGE-L
│   └── rapport_eval.json           # Rapport d'evaluation genere
├── monitoring/
│   ├── analyse_derive.py           # Analyse Evidently AI
│   ├── prometheus.yml              # Configuration Prometheus
│   ├── alertes.yml                 # Regles d'alerte
│   ├── docker-compose.yml          # Prometheus + Grafana
│   └── grafana_dashboard.json      # Dashboard Grafana exporte
├── prompts/
│   └── default_prompt.txt          # Prompt systeme local
├── scripts/
│   └── check_model_weights.py      # Hook anti-poids de modele
├── src/
│   ├── modele.py                   # Chargement/generation LLM
│   ├── prompts.py                  # Gestion des prompts
│   ├── tokeniseur.py               # Fonctions de tokenisation
│   └── rag/
│       ├── base_connaissance.py    # Creation base vectorielle
│       ├── recherche.py            # Recherche semantique
│       ├── chunking.py             # Decoupage documents longs
│       ├── reranking.py            # Reclassement cross-encoder
│       ├── pipeline.py             # Pipeline RAG complet
│       └── local_backend.py        # Backend demo/test en memoire
├── tests/                          # Tests unitaires
├── .github/workflows/              # Workflows CI GitHub Actions
├── pyproject.toml                  # Configuration outils Python
├── requirements.txt                # Dependances applicatives
├── requirements_tests.txt          # Dependances tests/qualite
├── requirements_eval.txt           # Dependances evaluation optionnelle
└── Dockerfile
```

## Prerequis

- Python 3.10 ou plus recent.
- Windows, Linux ou macOS.
- Environ 8 Go de RAM pour charger le modele Qwen2.5-1.5B-Instruct en mode reel.
- Docker Desktop optionnel pour Prometheus/Grafana.
- Compte Langfuse optionnel pour les traces cloud.

Le projet fonctionne aussi en **mode demonstration**, sans telecharger de modele lourd. Ce mode est utilise par defaut pour les tests et les demos rapides.

## Installation

### 1. Cloner le depot

```bash
git clone <url-du-depot>
cd Projet
```

### 2. Creer et activer l'environnement virtuel

Sous PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sous Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dependances

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements_tests.txt
python -m pip install -r requirements_eval.txt
```

## Configuration

Copier le fichier d'exemple:

```bash
cp .env.example .env
```

Sous PowerShell:

```powershell
Copy-Item .env.example .env
```

Variables principales:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

LLMOPS_REAL_MODE=0
LLMOPS_LOAD_LLM=0
LLMOPS_LOAD_RERANKER=0
```

Par defaut:

- `LLMOPS_REAL_MODE=0`: mode demo rapide.
- `LLMOPS_LOAD_LLM=0`: pas de chargement du LLM lourd.
- `LLMOPS_LOAD_RERANKER=0`: pas de chargement du cross-encoder reel.

Pour utiliser ChromaDB et les modeles reels:

```env
LLMOPS_REAL_MODE=1
LLMOPS_LOAD_LLM=1
LLMOPS_LOAD_RERANKER=1
```

## Lancer l'API

```bash
uvicorn api.app:app --reload
```

L'API est disponible sur:

```text
http://127.0.0.1:8000
```

Documentation interactive Swagger:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

| Methode | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Etat du service et informations de configuration |
| POST | `/generate` | Generation d'une reponse avec le pipeline RAG |
| GET | `/metrics` | Metriques Prometheus |

### Exemple de requete

Recommande sous PowerShell:

```powershell
$body = @{
  prompt = "Sous quel delai puis-je retourner un article ?"
  nb_tokens_max = 120
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/generate" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Alternative via Swagger:

1. Ouvrir `http://127.0.0.1:8000/docs`.
2. Aller sur `POST /generate`.
3. Cliquer sur `Try it out`.
4. Envoyer:

```json
{
  "prompt": "Sous quel delai puis-je retourner un article ?",
  "nb_tokens_max": 120
}
```

Exemple de reponse:

```json
{
  "reponse": "Vous pouvez retourner tout article dans un delai de 30 jours suivant la reception de votre commande.",
  "nb_tokens": 17,
  "sources": ["faq-01", "faq-04", "faq-13"],
  "hors_domaine": false,
  "duree_ms": 4.21,
  "trace_url": "https://cloud.langfuse.com/project/.../traces/..."
}
```

## Observabilite Langfuse

Si les cles Langfuse sont configurees, chaque appel a `/generate` cree une trace `rag-generate`.

La trace contient les observations suivantes:

```text
rag-generate
├── retrieval
├── reranking
└── llm
```

Details traces:

- `retrieval`: prompt utilisateur, documents retrouves, scores de similarite.
- `reranking`: candidats initiaux, ordre final, scores de reranking.
- `llm`: prompt augmente, reponse generee, usage tokens.

Des scores sont aussi attaches a la trace:

- `nb_docs_retrouves`
- `score_reranking_max`
- `longueur_reponse_mots`
- `hors_domaine`

Le champ `trace_url` dans la reponse API permet d'ouvrir directement la trace dans Langfuse.

## Evaluation du RAG

Le jeu d'evaluation se trouve dans:

```text
evaluation/jeu_evaluation.jsonl
```

Lancer l'evaluation:

```bash
python evaluation/evaluer_rag.py
```

Lancer avec seuils bloquants:

```bash
python evaluation/evaluer_rag.py --fail-on-threshold
```

Metriques calculees:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `MRR`
- `ROUGE-L moyen`

Le rapport est genere dans:

```text
evaluation/rapport_eval.json
```

## Tests et qualite

Lancer les tests:

```bash
python -m pytest
```

Lancer la couverture:

```bash
python -m coverage run -m pytest
python -m coverage report -m
```

Lancer Ruff:

```bash
python -m ruff check .
```

Installer les hooks pre-commit:

```bash
pre-commit install
pre-commit run --all-files
```

Le hook local `scripts/check_model_weights.py` bloque les fichiers de poids de modele:

- `.pt`
- `.bin`
- `.safetensors`
- `.onnx`

## Monitoring Prometheus et Grafana

Lancer la stack de monitoring:

```bash
docker compose -f monitoring/docker-compose.yml up
```

Services:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Metriques suivies:

- latence HTTP;
- duree du retrieval;
- duree du reranking;
- duree de generation LLM;
- nombre de documents retrouves;
- nombre de tokens generes.

Le dashboard Grafana est disponible dans:

```text
monitoring/grafana_dashboard.json
```

## Detection de derive

Le script Evidently compare les donnees de reference et les logs de production:

```bash
python monitoring/analyse_derive.py
```

Sorties generees:

```text
monitoring/rapport_derive.html
monitoring/rapport_derive.json
```

La derive peut signaler:

- augmentation des requetes hors domaine;
- prompts plus longs;
- baisse du nombre de documents RAG retrouves;
- changement de vocabulaire utilisateur.

## Integration continue

Le depot contient trois workflows GitHub Actions:

| Workflow | Role |
| --- | --- |
| `codecheck.yaml` | Lance pre-commit, Ruff, mypy et controles de securite |
| `tests.yaml` | Lance les tests sur Linux/Windows et plusieurs versions Python |
| `rag_eval.yaml` | Lance l'evaluation RAG planifiee ou manuelle |

Ces workflows permettent de detecter rapidement:

- regressions fonctionnelles;
- baisse de couverture;
- degradation de qualite RAG;
- erreurs de style ou de typage;
- fichiers sensibles ou trop volumineux.

## Documentation projet

| Fichier | Description |
| --- | --- |
| `docs/guide_pas_a_pas.md` | Explication progressive du fonctionnement |
| `docs/rapport.md` | Rapport technique complet |
| `docs/soutenance.md` | Plan de soutenance |
| `sujet_TP2_json_schema.pdf` | Sujet original du TP |

## Resultats actuels

Derniere evaluation en mode demo:

```json
{
  "recall_at_1": 0.8,
  "recall_at_3": 0.9333,
  "recall_at_5": 0.9333,
  "mrr": 0.8667,
  "rouge_l_moyen": 0.6095,
  "nb_questions": 18
}
```

Couverture de tests:

```text
77 %
```

## Limites connues

- Le mode demo utilise un backend leger pour garantir des tests rapides.
- Le mode reel peut demander plusieurs Go de RAM selon le modele charge.
- ROUGE-L reste une metrique lexicale: elle ne mesure pas parfaitement la fidelite semantique.
- L'evaluation Ragas est optionnelle et depend d'un LLM juge.
- Le prompt Langfuse doit etre cree dans l'interface pour remplacer le fallback local.

## Ameliorations possibles

- Ajouter une interface web simple pour interroger l'assistant.
- Ajouter un vrai jeu de donnees metier plus volumineux.
- Integrer Ragas dans la CI pour des metriques semantiques.
- Ajouter des alertes Grafana connectees a un canal de notification.
- Deployer l'API sur un serveur cloud ou une plateforme containerisee.

## Auteur

Projet realise par Vincent dans le cadre d'un module LLMOps.

