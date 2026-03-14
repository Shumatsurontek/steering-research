# Steering LLM Research — Plan d'Action

## Objectif
Comparer **steering vectors** vs **prompt engineering** dans un contexte agentique
(chatbot de prise de rendez-vous calendrier), avec analyse comparative multi-modèles.

---

## Phase 1 — Fondations & Analyse des Tokenizers ✅
> Comprendre comment chaque modèle découpe les instructions agentiques

### 1.1 Setup environnement ✅
- [x] Environnement Python (venv), dépendances (transformers, torch, safetensors, einops)
- [x] Téléchargement Qwen3-4B-Instruct-2507 via HuggingFace

### 1.2 Analyse comparative des tokenizers ✅
- [x] Comparer les tokenizers de 4 modèles (Qwen3-4B, Gemma-3-1B, Llama-3.2-3B, Phi-3-mini)
- [x] Métriques : taille vocabulaire, compression ratio, découpage des instructions agentiques
- [x] Focus : comment chaque tokenizer traite les tool calls JSON, les dates, les noms propres
- [x] Visualisations comparatives pour l'article (5 plots)
- [x] Quality scoring function (temporal integrity, semantic coherence, JSON efficiency, multilingual parity)

**Résultats clés :**
- Llama-3.2-3B domine (overall: 0.551) — meilleure compression JSON et temporelle
- Tous les tokenizers fragmentent les timestamps ISO 8601 (temporal integrity < 0.32)
- Parité FR/EN correcte (~0.80-0.85) sur tous les modèles

---

## Phase 2 — Interprétabilité & Extraction de Features ✅
> Identifier les circuits internes pertinents pour le task scheduling

### 2.1 SAE-inspired contrastive analysis ✅
- [x] Hook-based activation extraction aux 36 couches de Qwen3-4B
- [x] 10 prompts calendrier vs 10 prompts neutres (contrastive pairs)
- [x] Vecteurs de steering = mean difference par couche (last-token position)
- [x] Layer importance ranking par L2 norm et cosine distance
- [x] Logit lens sur les top-3 couches (layers 33-35)

**Résultats clés :**
- Couche 35 = plus discriminative (L2 norm: 361.0, cosine dist: 0.151)
- Logit lens confirme : `schedule`, `agenda`, `attendees`, `RSVP`, `calendar` promus
- Signal cross-lingue détecté (tokens chinois email/modify promus)

### 2.2 Neuronpedia feature mapping ✅
- [x] Exploration des features SAE via API Neuronpedia (transcoder-hp, 163,840 features/couche)
- [x] 3 stratégies : search by explanation, random sampling, key layer scan
- [x] 115 features calendrier via search, 9/450 via sampling (densité 0.07%)
- [x] Features distribuées layers 20-35, spécialisation croissante

---

## Phase 3 — Steering Vectors ✅
> Construire et tester des vecteurs de steering pour le task calendrier

### 3.1 Last-layer steering (initial) ✅
- [x] Sweep de coefficients (α = 0, 5, 15, 30, 50) à la couche 35
- [x] **Résultat : INEFFICACE.** Réponses identiques à tous les coefficients.

### 3.2 Mid-layer steering sweep ✅
- [x] Sweep sur 11 couches × 6 coefficients (α = 0, 10, 30, 60, 100, 200)
- [x] **Résultat clé : layers 15-18 à α=30 = 100% change rate, réponses cohérentes**
- [x] Gradient de rigidité : early=instable, mid=sweet spot, late=rigide
- [x] Non-calendar prompts réinterprétés comme tâches calendrier

### 3.3 Base model steering ✅
- [x] Steering sur Qwen3-4B (base, non-instruct) avec vecteurs contrastifs frais
- [x] **Sweet spot identique : layer 15, α=30 → cal_score 0.70→0.97 (+0.27)**
- [x] **Base model plus fragile : dégénère à α≥60 (score→0.0)**
- [x] Rigidité late-layer confirmée comme propriété architecturale (pas instruction tuning)

### 3.4 Budget guidance ✅
- [x] Implémentation Gamma-distribution predictor (inspiré arxiv:2506.13752)
- [x] Test sur 5 budgets (32, 64, 128, 256, 512 tokens)
- [x] **Résultat négatif : 0% savings** — le modèle instruct produit déjà des outputs compacts (~75 tokens)
- [x] Budget guidance cible le thinking overhead, pas la verbosité d'output structuré

---

## Phase 4 — Standardized Benchmarks ✅ (partial)
> Évaluation quantitative du steering sur des benchmarks standardisés

### 4.1 GSM8K (Qwen3-0.6B) ✅
- [x] Test SLM (Qwen3-0.6B) sur GSM8K vs prompt engineering
- [x] Sampling-based analysis (T>0, KL divergence, diversity)
- [x] Validation lm-eval GSM8K (n=50) — standardized 5-shot evaluation

**Résultats clés SLM GSM8K (pilot n=10) :**
- Instruct : zero-shot=20%, best steering=30% (L25@α=60, +10%)
- Base : few-shot=20%, best steering=40% (L20@α=100, +20%)
- Sweet spot à 64-89% depth (vs 42-50% sur 4B) → relative depth hypothesis
- CoT hurts 0.6B (10% < 20% zero-shot)

**Résultats clés Sampling :**
- KL divergence mid-layer : 1-48 bits vs late-layer : <0.01 bits
- Diversity L15@α=30 : 80-100% (vs baseline 20-40%)
- Rigidité late-layer confirmée distributionnellement
- Output JSON préservé malgré diversité accrue

**Résultats lm-eval GSM8K (n=50) — 5-shot vs zero-shot CoT :**

| Modèle | Condition | 5-shot strict | 5-shot flex | 0-shot strict | 0-shot flex |
|--------|-----------|:---:|:---:|:---:|:---:|
| Instruct | Baseline | 48% | 48% | 38% | 46% |
| Instruct | Steered | 44% (-4) | 40% (-8) | 40% (+2) | **62% (+16)** ★ |
| Base | Baseline | 48% | 48% | 36% | 28% |
| Base | Steered | 26% (-22) | 34% (-14) | 8% (-28) | 22% (-6) |

- **Zero-shot + steering = synergie** (+16pp instruct flexible) — meilleur résultat global
- **5-shot + steering = interférence destructrice** — les exemplaires few-shot et le vecteur sont redondants
- **Conclusion clé : α = f(n_few_shot, model_type)** — coefficient adaptatif nécessaire
- Script : `src/steering/gsm8k_benchmark.py`

### 4.2 MMLU-Pro Multi-Model Benchmark (Next — Final Experiment)
> Hypothèse : le steering domain-specific au zero-shot améliore les SLMs sur un benchmark multi-domaine exigeant

**Dataset : TIGER-Lab/MMLU-Pro**
- 12,032 questions, 14 domaines (math, physics, law, CS, biology, etc.)
- 10 options (vs 4 MMLU) → baseline chance = 10%
- CoT critique (+20% vs direct) — le steering en zero-shot est la condition idéale
- Déjà dans lm-eval : `--tasks mmlu_pro`

**Modèles testés (3) :**

| Modèle | Params | Architecture | Intérêt |
|--------|--------|-------------|---------|
| Qwen3-0.6B | 0.6B | Transformer (28 layers) | Référence steering, déjà calibré |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | Hybrid SSM+Attention (16 layers) | Architecture non-standard, test de généralisation du steering |
| Llama-3.2-3B-Instruct | 3B | Transformer (32 layers) | Étudié en Phase 1 (tokenizer), scaling test |

**Plan d'expériences :**
- [ ] Extraction de vecteurs domain-specific pour chaque modèle (14 domaines MMLU-Pro)
- [ ] Sweep layer × α par domaine pour identifier les sweet spots
- [ ] lm-eval MMLU-Pro : baseline vs steered (zero-shot) par domaine × modèle
- [ ] Analyse : quels domaines bénéficient le plus du steering ? Est-ce que le sweet spot relatif (% depth) est constant cross-model ?
- [ ] Test cross-architecture : le steering fonctionne-t-il sur SSM+Attention (LFM2.5) ?

**Questions de recherche :**
- Le steering domain-specific améliore-t-il les SLMs au-delà de ce que CoT apporte ?
- La profondeur relative du sweet spot est-elle un invariant architectural ?
- Les architectures hybrides SSM/Attention répondent-elles au steering de la même manière que les transformers purs ?

---

## ~~Phase 5 — Calendar Prompt Engineering~~ (Archivé)
> Résultats préliminaires non concluants — remplacé par MMLU-Pro multi-model

- [x] 29 cas de test bilingues, 5 stratégies de prompting conçues
- [ ] ~~Inference Ollama/llama.cpp~~ — abandonné au profit de benchmarks standardisés
- Voir `src/prompts/` et `data/calendar_test_cases.json` pour référence

---

## Phase 6 — Dynamic Steering for Multi-Agent Orchestration (Next)
> Hypothèse : injecter dynamiquement des steering vectors domain-specific à chaque étape d'un plan agentique pour améliorer les performances SLM sur des tâches complexes (SWE-bench)

### 6.1 Concept — "Steering-as-a-Skill"
L'orchestrateur agentique identifie le domaine de chaque tool call (code, math, debugging, NL reasoning) et bind le steering vector correspondant via `register_forward_hook` avant l'inférence. Le modèle reste identique, seul le biais directionnel change dynamiquement.

```
┌─────────────────────────────────────────────────────┐
│              AGENT ORCHESTRATOR                      │
│                                                      │
│  Plan: [read_file] → [analyze_bug] → [write_patch]  │
│            │              │               │          │
│            ▼              ▼               ▼          │
│    ┌──────────┐   ┌────────────┐   ┌───────────┐    │
│    │ v_code   │   │ v_debug    │   │ v_code    │    │
│    │ L18@α=30 │   │ L20@α=30  │   │ L18@α=30  │    │
│    └────┬─────┘   └─────┬──────┘   └─────┬─────┘    │
│         ▼               ▼               ▼            │
│    ┌─────────────────────────────────────────┐       │
│    │         SLM (Qwen3-0.6B / 4B)           │       │
│    │   hook(layer_n) += α · v_domain/‖v‖     │       │
│    └─────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 6.2 Extraction de vecteurs domain-specific ✅
- [x] 4 domaines génériques : code_reading, bug_analysis, patch_writing, test_reasoning
- [x] 3 clusters SWE-bench : django_web, scientific_computing, dev_tooling
- [x] Sweet spot par domaine identifié (L15-18 génériques, L18-25 clusters)
- [x] Steering library sauvegardée : `domain_steering_vectors.pt` + `swebench_cluster_vectors.pt`

**Résultats clés :**
- Chaque domaine a un sweet spot distinct (L15@α=30 vs L18@α=60 vs L25@α=10)
- test_reasoning = le plus steerable (score 11), django_web = meilleur cluster (score 13)
- Cosine similarity clusters vs génériques : 0.84-0.87 (signal additionnel capturé)

### 6.3 Composition et interférence ✅
- [x] Addition : pas de dégénérescence MAIS dilution du signal (score 4.0→2.0)
- [x] Weighted 0.7/0.3 : meilleur que 0.5/0.5 mais toujours sous baseline
- [x] Sequential switching : clean, pas de résiduel entre hooks
- [x] **Verdict : sequential switching >> composition simultanée**

### 6.4 Orchestrateur dynamique ✅
- [x] Prototype `steering_orchestrator.py` avec hook switching par étape
- [x] 3 scénarios SWE-bench × 3 variantes (baseline, static, dynamic)
- [x] **Dynamic = 4-7× keyword hits vs baseline, 100% coherence, -20% tokens**
- [x] Static steering *pire* que baseline sur tâches hétérogènes

### 6.5 Benchmark SWE-bench Verified
- [x] Dataset analysé : 500 instances, 12 repos, 71% bug fixes
- [x] Pipeline sans RAG (n=20) : baseline 75%, static 70%, dynamic 75% valid diffs — mais 0% resolved (paths inventés)
- [x] Module RAG (`swebench_rag.py`) : clone repo@base_commit, keyword extraction, file search/ranking, context injection
- [x] Intégration RAG dans pipeline (`--rag` flag) : 3 variantes rag_baseline, rag_static, rag_dynamic
- [x] Validation de paths : le modèle utilise désormais des vrais chemins de fichiers
- [ ] Évaluation RAG complète (n=20) avec Docker harness
- [ ] Comparer : SLM baseline vs SLM + dynamic steering vs SLM + RAG + steering
- [ ] Métriques : % resolved, patch quality (path validity), token efficiency

**Résultats préliminaires sans RAG (n=20) :**
- 0/20 patches appliquées — le modèle 0.6B invente des paths (ex: `core.validators.URLValidator.py`)
- Valid diffs : baseline 75%, static 70%, dynamic 75% (syntaxiquement corrects mais paths faux)

**RAG pilot (n=2) :**
- Paths réels utilisés (validation paths:1/1 sur instance matplotlib)
- Pipeline fonctionnelle : clone → keyword extraction → file ranking → context injection → generation

**RAG generation (n=20, thinking disabled) :**

| Variante | Valid Diffs | Path Validity | Avg Quality | Temps |
|----------|:----------:|:------------:|:-----------:|:-----:|
| rag_baseline | **95%** | **68.4%** | **0.655** | 519s |
| rag_static | 100% | 15.0% | 0.276 | 673s |
| rag_dynamic | 90% | 0.0% | 0.205 | 244s |

- **Le steering dégrade les patches RAG** : même pattern que GSM8K zero-shot/few-shot
- RAG = contexte implicite (comme few-shot) → steering + RAG = interférence destructrice
- Le modèle non steered (rag_baseline) est le meilleur → **steering optimal en régime zero-shot uniquement**
- Thinking mode de Qwen3 + steering = boucles infinies → `enable_thinking=False` obligatoire

### 6.6 Questions résolues et ouvertes
- **Composabilité** : ✅ Possible sans dégénérescence, mais dilue le signal → ne pas utiliser
- **Fenêtre de stabilité** : ✅ Chaque domaine a son propre sweet spot (layer ET α)
- **Scaling** : ❓ À tester sur 4B+ (le 0.6B instruct répond bien, le base moins)
- **Analogie LoRA** : ❓ Steering = "LoRA sans entraînement" mais avec less control — limites à explorer

---

## Phase 7 — Rédaction de l'Article ✅
> Synthèse des résultats

- [x] Article LaTeX ~12 pages, format arxiv-ready
- [x] Sections : Introduction, Related Work, Tokenizer Analysis, Feature Extraction, Steering (mid-layer + base), Budget Guidance, Neuronpedia, GSM8K, Sampling, Baselines, Discussion
- [x] Résultats intégrés : mid-layer sweet spot, base model fragility, budget guidance null, Neuronpedia sparsity, GSM8K SLM, KL divergence
- [x] Compilé en PDF (`article/main.pdf`)
- [x] Repo GitHub : https://github.com/Shumatsurontek/steering-research

---

## Stack Technique

```
Python 3.14 + venv
├── transformers 5.3.0    # Chargement modèles & tokenizers
├── torch 2.10.0          # Compute (MPS/CUDA/CPU)
├── safetensors 0.7.0     # Chargement poids SAE
├── einops 0.8.2          # Opérations tensorielles multi-couches
├── matplotlib 3.10.8     # Visualisations
├── pandas 3.0.1          # Data manipulation
├── huggingface-hub 1.6.0 # Model downloads
└── lm-eval              # EleutherAI standardized evaluation (GSM8K benchmark)
```

## Modèles

| Modèle | Params | Architecture | Usage |
|--------|--------|-------------|-------|
| Qwen3-4B-Instruct-2507 | 4B | Transformer | Modèle principal, steering instruct |
| Qwen3-4B | 4B | Transformer | Steering base model (comparaison) |
| Qwen3-0.6B | 0.6B | Transformer (28L) | SLM instruct : GSM8K, SWE-bench, MMLU-Pro |
| Qwen3-0.6B-Base | 0.6B | Transformer (28L) | SLM base : GSM8K steering |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | Hybrid SSM+Attention (16L) | MMLU-Pro cross-architecture steering |
| Llama-3.2-3B-Instruct | 3B | Transformer (32L) | Tokenizer + MMLU-Pro scaling test |
| Gemma-3-1B-IT | 1B | Transformer | Comparaison tokenizer |
| Phi-3-mini-4k-instruct | 3.8B | Transformer | Comparaison tokenizer |
