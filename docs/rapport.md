# Rapport technique - Projet LLMOps RAG

## 1. Introduction

Ce projet implémente un assistant conversationnel de service client reposant sur le principe du RAG, Retrieval-Augmented Generation. L'objectif n'est pas seulement de générer une réponse, mais de mettre en place une chaîne LLMOps complète: qualité de code, intégration continue, API, observabilité, monitoring, évaluation et détection de dérive.

Le cas d'usage est volontairement concret. Une organisation possède une FAQ, des politiques de retour, des fiches de livraison et des règles support. Le modèle de langage ne connaît pas ces informations par défaut. Le RAG permet de rechercher les passages utiles dans la base de connaissance avant de demander au LLM de répondre.

## 2. Architecture générale

Le pipeline suit l'enchaînement suivant:

```text
Question utilisateur
-> embedding de la question
-> recherche vectorielle dans la base
-> reranking des passages candidats
-> construction d'un prompt augmenté
-> génération de la réponse
-> traces, logs, métriques et scores
```

Les composants principaux sont:

- ChromaDB pour la base vectorielle persistante;
- sentence-transformers pour les embeddings;
- un cross-encoder pour le reranking;
- Qwen2.5-1.5B-Instruct comme LLM local recommandé;
- FastAPI pour exposer le pipeline;
- Langfuse pour l'observabilité LLM;
- Evidently AI pour la dérive;
- Prometheus et Grafana pour le monitoring système;
- pre-commit et GitHub Actions pour la qualité automatisée.

Le dépôt inclut aussi un mode démonstration. Ce mode utilise un backend vectoriel mémoire et un modèle d'embedding symbolique afin que les tests puissent tourner rapidement sans téléchargement de gros modèles.

## 3. Base de connaissance

La base de connaissance principale se trouve dans `data/faq_service_client.jsonl`. Chaque ligne contient un identifiant, une question, une réponse et une catégorie.

Le module `src/rag/base_connaissance.py` charge ce fichier, transforme chaque entrée en document textuel puis calcule son embedding. En production locale, la collection est stockée dans `data/chroma_db/` via `chromadb.PersistentClient`.

Le choix de la distance cosinus est adapté aux embeddings normalisés. Deux textes proches sémantiquement ont une distance faible et donc une similarité élevée après conversion par `similarité = 1 - distance`.

## 4. Recherche sémantique

La fonction `_rechercher_documents()` encode la requête utilisateur puis interroge la collection. Le résultat contient les documents, les métadonnées, les IDs et les distances.

Un seuil de similarité est appliqué. Ce seuil est important car il évite d'injecter des documents non pertinents dans le prompt. Sans seuil, une question hors domaine obtiendrait malgré tout les documents "les moins mauvais", ce qui pousserait le LLM à produire une réponse fausse.

Le comportement attendu est donc:

- requête pertinente: documents sources retrouvés;
- requête hors domaine: aucun document au-dessus du seuil;
- réponse finale: `Je n'ai pas l'information.`

## 5. Prompt augmenté

Le prompt local est dans `prompts/default_prompt.txt`. Il impose trois règles:

- l'assistant est un assistant de service client;
- il doit répondre uniquement à partir du contexte;
- si l'information manque, il doit refuser de répondre.

Cette structure distingue clairement le contexte et la question. Cela réduit le risque que le modèle confonde les instructions, les sources et la demande de l'utilisateur.

## 6. Chunking

Les documents longs sont traités par `src/rag/chunking.py`. Le découpage est basé sur les mots avec deux paramètres:

- `taille_chunk`: nombre maximal de mots par chunk;
- `chevauchement`: nombre de mots repris entre deux chunks consécutifs.

Le chunking améliore la recherche car un document long parle souvent de plusieurs sujets. L'embedding d'un document entier devient une moyenne sémantique imprécise. Avec des chunks, chaque passage porte un sujet plus clair.

Le chevauchement évite de perdre une information coupée à la frontière de deux chunks. En revanche, trop de chevauchement augmente le nombre de chunks, le coût d'indexation et le bruit potentiel.

## 7. Reranking

La recherche initiale repose sur un bi-encoder. Il est rapide car les embeddings des documents sont pré-calculés. Sa limite est qu'il compare deux vecteurs globaux sans relire finement l'interaction entre la question et le passage.

Le reranking utilise un cross-encoder. Il reçoit chaque paire `(question, passage)` et produit un score plus précis. Comme il est plus lent, il ne doit pas être appliqué à toute la base, mais seulement aux candidats du bi-encoder.

Dans ce projet, la stratégie est:

1. retrieval top-10 avec bi-encoder;
2. reranking des 10 candidats;
3. injection du top-3 final dans le prompt.

## 8. API FastAPI

L'API est définie dans `api/app.py`. Elle expose:

- `GET /health`;
- `POST /generate`;
- `GET /metrics` si Prometheus est installé.

Les ressources lourdes sont chargées au démarrage. C'est indispensable pour éviter de recharger Qwen, ChromaDB ou le cross-encoder à chaque requête. Le chargement par requête dégraderait fortement la latence et rendrait le service instable.

Chaque appel `/generate` produit une réponse JSON avec la réponse, le nombre de tokens, les sources utilisées, l'indicateur hors domaine et la durée.

## 9. Observabilité avec Langfuse

Langfuse apporte une visibilité spécifique aux applications LLM. Prometheus indique qu'une requête a pris 6 secondes. Langfuse permet de savoir si ces 6 secondes viennent du retrieval, du reranking, de la génération ou d'un prompt trop long.

Chaque requête peut créer une trace `rag-generate` contenant:

- span `retrieval`;
- span `reranking`;
- generation `llm`;
- scores `nb_docs_retrouves`, `score_reranking_max`, `longueur_reponse_mots`, `hors_domaine`.

Le prompt peut aussi être géré dans Langfuse Prompt Management. L'intérêt est de modifier et versionner un prompt sans redéployer immédiatement le code.

## 10. Monitoring et dérive

Le monitoring système repose sur Prometheus et Grafana. Les métriques suivies sont:

- latence HTTP;
- durée de recherche RAG;
- durée de génération LLM;
- nombre de documents retrouvés;
- tokens générés.

La dérive est analysée avec `monitoring/analyse_derive.py`. Le script compare un corpus de référence à des logs de production. Une dérive peut apparaître si:

- les prompts deviennent plus longs;
- de nouveaux sujets apparaissent;
- le nombre de documents RAG retrouvés baisse;
- le taux hors domaine augmente.

Cette surveillance est complémentaire de Langfuse. Evidently observe les distributions de données, tandis que Langfuse explique les traces individuelles.

## 11. Évaluation RAG

Le jeu `evaluation/jeu_evaluation.jsonl` contient des questions, des réponses de référence et les documents pertinents attendus.

Les métriques utilisées sont:

- Recall@k: vérifie si un document pertinent apparaît dans les k premiers résultats;
- MRR: mesure le rang du premier document pertinent;
- ROUGE-L: mesure le recouvrement lexical entre réponse générée et référence.

Le script `evaluation/evaluer_rag.py` génère `evaluation/rapport_eval.json` et peut échouer si les seuils ne sont pas atteints. Cela permet de transformer la qualité RAG en garde-fou de CI.

## 12. Qualité et intégration continue

pre-commit bloque les erreurs avant le commit:

- espaces en fin de ligne;
- YAML/JSON invalide;
- clés privées;
- fichiers trop lourds;
- lint et formatage Ruff;
- vérification mypy;
- poids de modèles interdits.

GitHub Actions complète ce contrôle avec:

- `codecheck.yaml`: qualité;
- `tests.yaml`: tests Linux/Windows, Python 3.10/3.11/3.12;
- `rag_eval.yaml`: évaluation planifiée.

La matrice OS x Python est utile car certains bugs n'apparaissent que sur un environnement. Exemple: les chemins Windows utilisent `\`, alors que Linux utilise `/`. Un script fragile sur les chemins peut passer sur Linux et échouer sur Windows.

## 13. Analyse critique

Le projet reste limité par la taille du LLM. Qwen2.5-1.5B-Instruct est raisonnable pour une machine locale, mais il peut ignorer une instruction, paraphraser maladroitement ou produire une réponse trop courte. Un modèle plus grand améliore généralement la fidélité au contexte, mais augmente le coût mémoire, la latence et la complexité de déploiement.

ROUGE-L est également limité. Il compare les mots, pas le sens. Deux réponses sémantiquement équivalentes peuvent obtenir un score faible si elles utilisent des formulations différentes. Des métriques comme faithfulness, answer relevance ou un LLM-as-a-judge via Ragas complètent mieux l'analyse.

Le seuil de similarité doit être calibré expérimentalement. Trop bas, il injecte du bruit. Trop haut, il refuse trop souvent de répondre. La bonne valeur dépend de la base, du modèle d'embedding et du niveau de risque acceptable.

## 14. Réponses aux questions de réflexion

### 1. Pourquoi encoder la question plutôt que la réponse ?

Dans une FAQ, l'utilisateur formule généralement une question. Encoder la question FAQ rapproche l'espace de recherche de la requête utilisateur. Si on encode seulement les réponses, on compare une question à une phrase déclarative, ce qui peut réduire la similarité. Encoder question et réponse ensemble est souvent un compromis robuste.

### 2. Comment choisir taille de chunk et chevauchement ?

La taille dépend de la fenêtre de contexte du LLM, de la granularité des documents et du coût de recherche. Des chunks courts améliorent la précision mais augmentent le nombre de vecteurs. Des chunks longs gardent plus de contexte mais peuvent diluer le sujet. Le chevauchement typique est 10 à 20 %. On l'évalue avec Recall@k et inspection qualitative des passages récupérés.

### 3. Pourquoi ne pas utiliser directement un cross-encoder sur toute la base ?

Le cross-encoder est trop lent à grande échelle car il doit inférer chaque paire question-document. Le bi-encoder pré-calcule les documents et cherche vite. Le cross-encoder devient utile seulement sur une petite liste de candidats.

### 4. Monitoring système vs observabilité LLM

Prometheus indique l'état technique: latence, débit, erreurs, ressources. Langfuse explique le comportement LLM: prompt exact, documents injectés, spans, génération et scores. Prometheus aide à décider de scaler ou corriger une panne. Langfuse aide à corriger un prompt, une base ou un modèle.

### 5. Risques des prompts dans le code source

Un prompt en dur nécessite commit, CI et redéploiement pour chaque modification. Cela ralentit l'expérimentation et rend les retours arrière plus lourds. Langfuse versionne les prompts, permet de choisir une version de production et garde l'historique des changements.

### 6. Que signifie une hausse du score hors domaine ?

Si le score hors domaine augmente, les utilisateurs posent probablement des questions non couvertes par la base actuelle. L'action corrective est d'analyser ces requêtes, enrichir la FAQ, créer de nouvelles catégories et recalibrer le seuil.

### 7. Limites de ROUGE

ROUGE ne mesure pas la vérité, la fidélité au contexte, la politesse, la sécurité ni le sens. Il mesure un chevauchement lexical. Les alternatives sont Ragas, LLM-as-a-judge, évaluations humaines, faithfulness, answer relevance et tests métier ciblés.

### 8. pre-commit vs CI

pre-commit donne un retour immédiat au développeur avant le commit. La CI vérifie dans un environnement propre et partagé. Les deux sont nécessaires: pre-commit accélère la boucle locale, la CI garantit que le dépôt reste fiable pour tous.

### 9. Valeur de tester Windows et Linux

La valeur vient des différences de chemins, encodage, shell, fins de ligne et dépendances système. Exemple concret: un script qui concatène des chemins avec `/` peut fonctionner sur Linux et échouer sur Windows.

### 10. Dérive des données vs dérive du modèle

La dérive des données signifie que les entrées changent. La dérive du modèle signifie que la relation entre entrées et sorties attendues se dégrade, même avec des entrées semblables. Le score hors domaine signale que les requêtes s'éloignent de la base. Evidently signale que les distributions changent. Ensemble, ils aident à distinguer manque de connaissance et changement de comportement.

### 11. Limites de Qwen2.5-1.5B-Instruct

Un petit modèle peut produire des répétitions, ignorer partiellement le contexte ou répondre de façon trop générique. Un modèle plus grand suit mieux les instructions et paraphrase mieux, mais coûte plus cher. Ragas révélerait la fidélité et la pertinence sémantique que ROUGE ne capture pas.

## 15. Conclusion

Ce projet montre qu'une application LLM fiable ne se limite pas au modèle. La valeur vient de la chaîne complète: base de connaissance, retrieval, reranking, prompts, API, traces, métriques, tests, CI et évaluation continue. Cette approche rend le système plus explicable, plus maintenable et plus défendable en production.

