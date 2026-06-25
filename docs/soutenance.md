# Plan de soutenance

## Slide 1 - Contexte

Objectif: déployer un assistant service client fiable. Le problème principal est que le LLM seul ne connaît pas les données internes et peut halluciner.

Message clé: le RAG ancre la réponse dans une base de connaissance contrôlée.

## Slide 2 - Architecture globale

Présenter le flux:

```text
Client -> FastAPI -> Retrieval -> Reranking -> Prompt augmenté -> LLM -> Réponse
                         |             |             |
                      ChromaDB      Cross-encoder   Langfuse
```

Ajouter les briques transverses: pre-commit, GitHub Actions, Evidently, Prometheus/Grafana.

## Slide 3 - Base de connaissance

Montrer `data/faq_service_client.jsonl`.

Expliquer:

- chaque ligne est une entrée FAQ;
- l'embedding représente le sens du document;
- ChromaDB permet la recherche vectorielle locale.

## Slide 4 - Recherche et seuil hors domaine

Exemple:

Question: "Sous quel délai puis-je retourner un article ?"

Résultat attendu: `faq-01`.

Point important: si la similarité est trop faible, on ne force pas le LLM à répondre. On renvoie une réponse de refus.

## Slide 5 - Chunking

Expliquer pourquoi on ne met pas un PDF entier dans le prompt:

- fenêtre de contexte limitée;
- embedding trop général;
- bruit dans la recherche.

La solution est le découpage avec chevauchement.

## Slide 6 - Reranking

Comparer:

- bi-encoder: rapide, utilisé sur toute la base;
- cross-encoder: précis, utilisé seulement sur les candidats.

Conclusion: le reranking améliore la qualité quand plusieurs passages sont proches.

## Slide 7 - API et observabilité

Montrer `/health`, `/generate`, `/metrics`.

Expliquer qu'une trace Langfuse contient:

- la requête;
- les documents retrouvés;
- le prompt augmenté;
- la réponse;
- les scores.

## Slide 8 - Monitoring et dérive

Prometheus/Grafana répond à: "Le service est-il rapide et disponible ?"

Evidently répond à: "Les requêtes utilisateurs ressemblent-elles encore au corpus attendu ?"

Langfuse répond à: "Quelle étape du pipeline explique le résultat observé ?"

## Slide 9 - Évaluation

Présenter les métriques:

- Recall@k: le bon document est-il retrouvé ?
- MRR: à quelle position arrive le premier bon document ?
- ROUGE-L: la réponse ressemble-t-elle à la référence ?

Insister sur la limite: ROUGE ne comprend pas le sens.

## Slide 10 - CI et qualité

Montrer les trois workflows GitHub Actions:

- qualité;
- tests multi-OS/multi-Python;
- évaluation RAG planifiée.

Expliquer pourquoi pre-commit évite de découvrir les erreurs trop tard.

## Slide 11 - Limites

Limites à assumer:

- un petit LLM peut ignorer le contexte;
- ROUGE est lexical;
- le choix du seuil de similarité doit être validé expérimentalement;
- Langfuse nécessite des clés ou un déploiement local;
- le mode démo ne remplace pas les vrais embeddings.

## Slide 12 - Conclusion

Le projet couvre toute la chaîne LLMOps:

- qualité avant commit;
- recherche augmentée;
- API;
- traces;
- monitoring;
- évaluation;
- détection de dérive.

Message final: l'objectif n'est pas seulement de faire répondre un modèle, mais de pouvoir mesurer, diagnostiquer et maintenir son comportement.

