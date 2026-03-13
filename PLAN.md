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

## Phase 4 — Prompt Engineering Baseline ✅
> Construire la baseline pour la comparaison

### 4.1 Stratégies de prompting ✅
- [x] Zero-shot, Few-shot (3 et 5 examples), Chain-of-thought, Tool use
- [x] System prompts avec date de référence pour les dates relatives
- [x] Format function calling compatible Ollama

### 4.2 Dataset et métriques ✅
- [x] 29 cas de test bilingues (18 FR, 11 EN) — 4 niveaux de complexité
- [x] Métriques : exact match, field accuracy (par champ), partial score (pondéré)
- [x] Framework d'agrégation des résultats prêt

**En attente :** Inference réelle avec Ollama/llama.cpp

---

## Phase 5 — Intégration Agentique (Future Work)
> Tester en conditions réelles avec Ollama/llama.cpp + LangChain

- [ ] Servir Qwen3-4B via Ollama
- [ ] Agent LangChain avec tool "create_calendar_event"
- [ ] Évaluation comparative sur les 29 cas avec les 5 stratégies
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
- Le steering améliore le raisonnement mais pas le formatage (flexible > strict)
- Base model à α=100 dégénère dans les deux cas → α trop agressif
- **Conclusion clé : α = f(n_few_shot, model_type)** — coefficient adaptatif nécessaire
- Script : `src/steering/gsm8k_benchmark.py` (subclasse HFLM avec hooks steering, `--task` flag)

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

### 6.2 Extraction de vecteurs domain-specific
- [ ] Définir les domaines SWE-bench : code_reading, bug_analysis, patch_writing, test_reasoning
- [ ] Construire des paires contrastives par domaine (10+ prompts positifs vs neutres)
- [ ] Extraire les vecteurs sur Qwen3-0.6B et Qwen3-4B
- [ ] Identifier le sweet spot (layer, α) par domaine via sweep automatisé
- [ ] Sauvegarder une "steering library" : `{domain: {layer, coeff, vector}}`

### 6.3 Composition et interférence
- [ ] Test d'addition de vecteurs : v_code + v_debug → cohérent ou dégénéré ?
- [ ] Test de switching séquentiel : v_code → remove → v_debug (pas de résiduel ?)
- [ ] Mesurer la fenêtre de stabilité par domaine (α_min, α_max avant dégénérescence)
- [ ] Explorer le coefficient adaptatif : α = f(confidence, task_complexity)

### 6.4 Benchmark SWE-bench Verified
- [ ] Charger le dataset : `SWE-bench/SWE-bench_Verified` (HuggingFace)
- [ ] Pipeline : repo clone → issue parsing → steering-augmented generation → patch apply → test
- [ ] Comparer : SLM baseline vs SLM + dynamic steering vs SLM + prompt engineering
- [ ] Métriques : % resolved, patch quality, token efficiency

### 6.5 Questions ouvertes
- **Composabilité** : peut-on additionner des vecteurs de domaines différents ?
- **Fenêtre de stabilité** : chaque domaine a-t-il son propre α_max ?
- **Scaling** : le gain relatif du steering est-il plus fort sur les SLMs (0.6B) que sur les LLMs (4B+) ?
- **Analogie LoRA** : steering vectors = "LoRA sans entraînement" — quelles sont les limites ?

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

| Modèle | Params | Usage |
|--------|--------|-------|
| Qwen3-4B-Instruct-2507 | 4B | Modèle principal, steering instruct |
| Qwen3-4B | 4B | Steering base model (comparaison) |
| Qwen3-0.6B | 0.6B | SLM instruct : GSM8K steering + lm-eval benchmark |
| Qwen3-0.6B-Base | 0.6B | SLM base : GSM8K steering + lm-eval benchmark |
| Gemma-3-1B-IT | 1B | Comparaison tokenizer |
| Llama-3.2-3B-Instruct | 3B | Comparaison tokenizer |
| Phi-3-mini-4k-instruct | 3.8B | Comparaison tokenizer |
