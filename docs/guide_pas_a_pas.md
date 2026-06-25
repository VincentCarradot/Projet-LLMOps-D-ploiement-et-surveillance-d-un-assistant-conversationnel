# Guide pas à pas du projet

## 1. Le problème à résoudre

Un LLM seul ne connaît pas les informations internes d'une entreprise. Il peut répondre de façon vague ou inventer. Le RAG corrige ce problème en ajoutant une étape de recherche avant la génération:

1. L'utilisateur pose une question.
2. La question est transformée en embedding.
3. La base vectorielle retrouve les passages proches.
4. Le prompt final contient la question et les passages retrouvés.
5. Le LLM répond uniquement à partir de ce contexte.

Dans ce projet, la base métier est la FAQ service client située dans `data/faq_service_client.jsonl`.

## 2. Base de connaissance

Le fichier `src/rag/base_connaissance.py` charge la FAQ, formate chaque entrée en document, calcule les embeddings et alimente une collection.

En mode réel, la collection est ChromaDB:

```python
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(
    name="faq_service_client",
    metadata={"hnsw:space": "cosine"},
)
```

En mode test, `src/rag/local_backend.py` remplace ChromaDB par une collection mémoire. Cela rend les tests rapides et reproductibles.

## 3. Recherche sémantique

La fonction centrale est `_rechercher_documents()` dans `src/rag/recherche.py`.

Elle fait trois choses:

1. Encode la requête utilisateur.
2. Interroge la collection avec `collection.query()`.
3. Convertit la distance cosinus en similarité avec `score = 1 - distance`.

Le seuil de similarité évite d'injecter du contexte faux. Pour une question hors domaine comme "quelle météo demain ?", le score reste trop faible et le pipeline répond: `Je n'ai pas l'information.`

## 4. Prompt augmenté

Le prompt est construit dans `src/rag/pipeline.py`.

Il sépare clairement:

- le rôle de l'assistant;
- le contexte;
- la question;
- la zone de réponse.

Cette séparation réduit les hallucinations, car le modèle voit explicitement ce qui est une source et ce qui est une demande.

## 5. Chunking

Le chunking est dans `src/rag/chunking.py`.

Un document long est découpé par fenêtre glissante:

```text
taille_chunk = 300 mots
chevauchement = 50 mots
pas = 250 mots
```

Le chevauchement garde un peu de contexte entre deux chunks voisins. Cela évite de couper une information importante à la frontière.

## 6. Reranking

Le bi-encoder est rapide mais approximatif. Il sert à récupérer un top-10.

Le cross-encoder est plus précis mais plus lent. Il relit chaque paire `(question, passage)` et reclasse les candidats. Le pipeline garde ensuite le top-3 final.

Dans le code:

- `src/rag/recherche.py`: retrieval rapide.
- `src/rag/reranking.py`: reranking précis.
- `src/rag/pipeline.py`: assemblage complet.

## 7. API FastAPI

L'API est dans `api/app.py`.

Endpoints:

- `GET /health`: vérifie que le service est vivant.
- `POST /generate`: lance le pipeline RAG.
- `GET /metrics`: expose les métriques Prometheus.

Les modèles sont chargés au démarrage, pas à chaque requête. C'est essentiel: un modèle comme Qwen2.5-1.5B peut prendre 10 à 30 secondes à charger.

## 8. Logs et observabilité

Chaque appel `/generate` écrit une ligne JSON dans `logs/requetes.jsonl` avec:

- prompt;
- réponse;
- durée;
- sources RAG;
- nombre de documents retrouvés;
- score de reranking;
- indicateur hors domaine.

Si Langfuse est configuré, l'API crée une trace par requête avec des spans:

- `retrieval`;
- `reranking`;
- `llm`.

Langfuse permet donc de comprendre pourquoi une réponse est lente ou mauvaise.

## 9. Évaluation RAG

Le script `evaluation/evaluer_rag.py` lit `evaluation/jeu_evaluation.jsonl` et calcule:

- `Recall@1`, `Recall@3`, `Recall@5`;
- `MRR`;
- `ROUGE-L moyen`.

Le retrieval mesure si le bon document est retrouvé. ROUGE-L mesure le recouvrement lexical entre la réponse générée et la réponse attendue.

## 10. Dérive

Le script `monitoring/analyse_derive.py` compare:

- `data/corpus_reference.jsonl`: requêtes normales;
- `logs/requetes.jsonl`: requêtes de production.

Si les prompts deviennent plus longs, hors domaine ou si le nombre de documents RAG retrouvés baisse, on peut détecter une dérive.

## 11. Monitoring système

Prometheus collecte les métriques exposées par FastAPI:

- latence HTTP;
- durée retrieval;
- durée génération;
- nombre de documents retrouvés;
- tokens générés.

Grafana affiche ces métriques dans `monitoring/grafana_dashboard.json`.

## 12. CI et qualité

Le dépôt contient trois workflows:

- `codecheck.yaml`: pre-commit, Ruff, formatage, mypy, sécurité.
- `tests.yaml`: matrice Linux/Windows et Python 3.10/3.11/3.12.
- `rag_eval.yaml`: évaluation planifiée du RAG.

Le hook local `scripts/check_model_weights.py` bloque les fichiers `.pt`, `.bin`, `.safetensors` et `.onnx` pour éviter de committer des poids de modèles.

