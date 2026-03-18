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

### 4.2 MMLU-Pro Multi-Model Benchmark ✅
> Hypothèse : le steering domain-specific au zero-shot améliore les SLMs sur un benchmark multi-domaine exigeant

**Dataset : TIGER-Lab/MMLU-Pro**
- 12,032 questions, 14 domaines, 10 options (baseline chance = 10%)
- 3 domaines sélectionnés par cosine dissimilarity : **math, law, history** (avg cos 0.249)

**Modèles testés (3) :**

| Modèle | Params | Architecture | Mid-layer |
|--------|--------|-------------|-----------|
| Qwen3-0.6B | 0.6B | Transformer (28L, 1024d) | L14 |
| Llama-3.2-3B-Instruct | 3B | Transformer (28L, 3072d) | L14 |
| LFM2.5-1.2B-Instruct | 1.2B | Hybrid SSM+Attention (16L, 2048d) | L8 |

**Dual-mode evaluation :**
- [x] `generate_until` (CoT) : Qwen complet, Llama math+law (history OOM)
- [x] `multiple_choice` (loglikelihood) : 3 modèles × 3 domaines × 4 conditions = 36 évals

**Résultats generate_until (n=20) — le steering dégrade systématiquement :**

| Modèle | Domaine | Baseline | α=10 | α=30 | α=60 |
|--------|---------|:--------:|:----:|:----:|:----:|
| Qwen3-0.6B | math | **30%** | 25% | 10% | 0% |
| Qwen3-0.6B | law | **15%** | 10% | 20% | 0% |
| Qwen3-0.6B | history | **20%** | 15% | 5% | 0% |
| Llama-3.2-3B | math | **25%** | 5% | 0% | 0% |
| Llama-3.2-3B | law | **30%** | 0% | 0% | 0% |

**Résultats loglikelihood (n=20) — pattern nuancé, améliorations ciblées :**

| Modèle | Domaine | Baseline | α=10 | α=30 | α=60 |
|--------|---------|:--------:|:----:|:----:|:----:|
| Qwen3-0.6B | math | 20% | **25%** | **25%** | 15% |
| Qwen3-0.6B | law | **20%** | 20% | 15% | 10% |
| Qwen3-0.6B | history | 5% | 10% | **20%** | **20%** |
| Llama-3.2-3B | math | **15%** | 10% | 5% | 5% |
| Llama-3.2-3B | law | 0% | **10%** | 5% | **10%** |
| Llama-3.2-3B | history | **20%** | 10% | 15% | 15% |
| LFM2.5-1.2B | math | **25%** | 10% | 5% | 5% |
| LFM2.5-1.2B | law | 10% | **15%** | **15%** | **15%** |
| LFM2.5-1.2B | history | **30%** | 10% | 10% | 10% |

**Finding clé n=20 (INVALIDÉ par n=200, voir ci-dessous) :**
- Qwen history : 5% → 20% (+15pp) semblait significatif mais stderr ~9pp ⇒ faux positif

**Validation n=200 (Qwen3-0.6B, loglikelihood) :**

| Domaine | Baseline | α=10 | α=30 | α=60 |
|---------|:--------:|:----:|:----:|:----:|
| math    | 25.5% (±3.1) | 27.0% (+1.5) | 24.0% (-1.5) | 20.0% (-5.5) |
| law     | 17.0% (±2.7) | 14.5% (-2.5) | 11.0% (-6.0) | 8.5% (-8.5) |
| history | 19.5% (±2.8) | 15.5% (-4.0) | 16.5% (-3.0) | 14.0% (-5.5) |

- **Steering dégrade systématiquement** toutes les conditions — aucune amélioration n'est significative
- stderr passé de ±9pp (n=20) à ±2.5pp (n=200) → résultats fiables
- history +15pp à n=20 → -4pp à n=200 : **faux positif classique de petit échantillon**

**Analyse qualitative per-sample :**
- Loglikelihood : le steering aplatit les distributions, crée un biais de position vers option A
- Generate-until : le steering remplace le raisonnement spécifique par du filler générique
- **Insight clé : "domain style vs domain knowledge"** — les vecteurs contrastifs capturent la saveur du domaine, pas les connaissances factuelles

**Scripts :** `mmlu_pro_vectors.py`, `mmlu_pro_benchmark.py`, `mmlu_pro_benchmark_mc.py`, `mmlu_pro_figures.py`, `mmlu_pro_samples.py`, `mmlu_pro_sample_figures.py`
**Custom tasks :** `src/steering/tasks/mmlu_pro_mc/` (loglikelihood mode)
**Figures :** `article/figures/mmlu_mc_*.pdf`, `article/figures/mmlu_samples_*.pdf`

### 4.3 Cross-Architecture Geometry Analysis ✅
> La géométrie des steering vectors est-elle universelle ou propre à chaque architecture ?

- [x] Comparaison des matrices cosine similarity 14×14 entre 3 modèles
- [x] Corrélation Spearman des upper triangles (91 paires de domaines)
- [x] MDS + Procrustes alignment pour projection 2D partagée
- [x] Figures publication-quality générées

**Résultats clés :**

| Paire de modèles | Spearman ρ | Pearson r |
|-------------------|:----------:|:---------:|
| Qwen3-0.6B vs Llama-3.2-3B | 0.893 | 0.920 |
| Qwen3-0.6B vs LFM2.5-1.2B | 0.936 | 0.957 |
| Llama-3.2-3B vs LFM2.5-1.2B | 0.888 | 0.909 |

- **La géométrie des domaines est invariante architecturalement** (ρ > 0.88, p < 10⁻³¹)
- History = outlier universel (avg cos le plus bas dans les 3 modèles)
- STEM (math, physics) cluster serré ; law et philosophy isolés
- Transfert direct impossible (dims: 1024, 3072, 2048) — projection linéaire nécessaire
- Normes L2 varient de 60× (Qwen ~17, Llama ~6, LFM2 ~0.25) → α doit être calibré par archi
- Procrustes disparity faible (~0.19-0.20) → topologie bien préservée

**Figures :** `article/figures/cross_model_mds.png`, `cross_model_combined.png`, `cross_model_heatmaps.png`
**Scripts :** `src/steering/cross_model_analysis.py`, `src/steering/cross_model_figures.py`
**Données :** `results/cross_model_analysis.json`

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

## Phase 7 — Rédaction de l'Article ✅ (itérations continues)
> Synthèse des résultats

- [x] Article LaTeX ~25 pages, format arxiv-ready
- [x] Sections : Introduction, Related Work, Tokenizer Analysis, Feature Extraction, Steering (mid-layer + base), Budget Guidance, Neuronpedia, GSM8K, Sampling, Baselines, Discussion (MMLU-Pro, cross-architecture, SAE, SWE-bench)
- [x] Résultats intégrés : mid-layer sweet spot, base model fragility, budget guidance null, Neuronpedia sparsity, GSM8K SLM, KL divergence, MMLU-Pro n=200 validation, per-sample qualitative analysis, SAE feature decomposition, cross-architecture geometry
- [x] 11 contributions listées dans l'introduction
- [x] Compilé en PDF (`article/main.pdf`)
- [x] Repo GitHub : https://github.com/Shumatsurontek/steering-research

---

## Phase 8 — SAE Feature Decomposition ✅
> Comprendre mécanistiquement pourquoi les vecteurs contrastifs échouent sur les benchmarks de connaissances

### 8.1 Entraînement SAE custom ✅
- [x] SAE Standard (SAELens) sur Qwen3-0.6B layer 14 (residual stream)
- [x] Config : d_in=1024, d_sae=8192 (8x expansion), L1=5e-3, lr=3e-4
- [x] Entraînement 5M tokens OpenWebText (MSE: 212K → 99)
- [x] Entraînement 20M tokens complété ✅ (MSE: 26.9, explained variance: 99.99%, L0=7131)
- [x] wandb : https://wandb.ai/arthur-edmond-perso/sae-qwen3-0.6b/runs/62p791yw
- [x] Intégration wandb pour visualisation des courbes de loss

**Scripts :** `src/steering/train_sae.py`
**Résultats :** `results/sae_qwen3_0.6b_L14_8x/` (cfg.json + sae_weights.safetensors, ~64MB)
**Note :** L0=7131/8192 (87% features actives) → L1 coefficient pourrait être augmenté pour plus de sparsité

### 8.2 Analyse des features par domaine ✅
- [x] Activation différentielle : 10 prompts × 3 domaines (math, law, history)
- [x] Forward pass → cache → SAE encode → mean per-prompt → top-k différentiel
- [x] Features spécialisées identifiées : differential activation 1.5-2.0 par domaine
- [x] Law features activent 3-4× plus fortement que math (mean 5-10 vs 2-4)

### 8.3 Comparaison contrastive vs SAE ✅
- [x] Projection des vecteurs contrastifs dans l'espace SAE via W_enc
- [x] **Chaque vecteur contrastif active 47-59% des 8192 features** (direction diffuse)
- [x] Math: 4664/8192 actives, overlap=2 ; Law: 4845/8192, overlap=**0** ; History: 3857/8192, overlap=4
- [x] Les 4 overlaps history sont des features humanities partagées law/history (5586, 5355, 8191)
- [x] **Conclusion : les vecteurs contrastifs ≠ features interprétables localisées**

**Implication :** Le "domain style vs domain knowledge" s'explique mécanistiquement :
- Features SAE domaine-spécifiques = sparse, localisées (top-20 sur 8192)
- Vecteurs contrastifs = directions diffuses activant la majorité du dictionnaire
- Pour du steering basé sur les connaissances : cibler des features SAE spécifiques plutôt que des moyennes contrastives

**Scripts :** `src/steering/analyze_sae_features.py`
**Données :** `results/sae_domain_analysis.json`

### 8.4 Feature-Targeted Steering Benchmark ✅
> Hypothèse : cibler les features SAE spécifiques au domaine via les colonnes du décodeur produit un steering plus précis

- [x] 3 stratégies : weighted (top-k × diff), uniform (top-k equal), single (best feature)
- [x] Vecteurs construits : v = Σ W_dec[feature_i] × weight_i pour top-k features
- [x] Benchmark MMLU-Pro MC (n=50) : 3 domaines × 3 stratégies × 3 coefficients (α=3,5,10)
- [x] Comparaison directe contrastive vs feature-targeted

**Résultats clés :**
- Feature uniform k20 à α=10 : meilleur sur math (22% vs 20% contrastive)
- Feature uniform k20 : **préserve la baseline** mieux que contrastive (law 24% maintenu vs 22%)
- À α≥30 : les deux approches dégradent — le modèle 0.6B n'a simplement pas les connaissances profondes
- **Conclusion : feature-targeted = dégradation plus douce, mais pas de gain significatif**

**Scripts :** `src/steering/feature_targeted_steering.py`
**Données :** `results/feature_targeted_benchmark_n50.json`

### 8.5 Entraînement SAE haute sparsité (L1=0.05) ✅
- [x] Lancement 20M tokens avec L1=0.05 (10× plus que baseline)
- [x] Analyse post-training : L0, overlap, features plus interprétables ?
- wandb : https://wandb.ai/arthur-edmond-perso/sae-qwen3-0.6b/runs/vgfylcxl

**Métriques L1=0.05 vs L1=0.005 :**
| Métrique | L1=0.005 | L1=0.05 |
|----------|:--------:|:-------:|
| MSE | 26.9 | 71.4 |
| Explained variance | 99.99% | 99.98% |
| L0 (features actives) | 7131 (87%) | 5793 (71%) |
| Contrastive activations | 47-59% | 48-56% |
| Overlap math | 2/20 | 3/20 |
| Overlap law | 0/20 | 0/20 |
| Overlap history | 4/20 | 0/20 |

**Conclusion :** 10× L1 réduit L0 de ~19% mais reste loin d'une sparsité interprétable (idéal : L0~50-200, soit <3%). Le finding principal (contrastive = diffus, domain = sparse) est robuste à la sparsité. Pour atteindre une vraie sparsité, il faudrait L1≥1.0 ou un SAE plus large (32K+ features).

### 8.6 Streamlit Demo ✅
- [x] App interactive : baseline vs contrastive vs feature-targeted en temps réel
- [x] Sidebar : domaine, coefficient, stratégie, top-k
- [x] Mode batch : 10 prompts par domaine
- [x] Diff visuel mot-par-mot avec coloration
- [x] Lancement : `streamlit run src/steering/app_steering_demo.py`

**Script :** `src/steering/app_steering_demo.py`

---

## Phase 9 — Qwen3-4B SAE Comparison 🔄
> Reproduire l'analyse SAE sur Qwen3-4B pour comparer avec Qwen3-0.6B : un modèle 6.7× plus grand montre-t-il des features plus interprétables et un meilleur steering feature-targeted ?

**Architecture Qwen3-4B :** 36 layers, hidden_dim=2560, SAE 8× = 20,480 features

### 9.1 Scripts rendus model-agnostiques ✅
- [x] `train_sae.py` : `--model`, `--layer`, `--d_in` avec presets par modèle
- [x] `analyze_sae_features.py` : `--model`, `--layer`, `--sae_dir`
- [x] `feature_targeted_steering.py` : `--model`, `--layer`
- [x] `mmlu_pro_vectors.py` : `--model` filter + Qwen3-4B ajouté
- [x] Tous les `stop_at_layer` dynamiques (plus de hardcoded 15)

### 9.2 Vecteurs contrastifs Qwen3-4B ✅
- [x] `python -m src.steering.mmlu_pro_vectors --model qwen3_4b`
- [x] Triplet le plus dissimilaire (L18) : physics, business, history (cos=0.335)
- [x] Fichier : `results/mmlu_pro_vectors_qwen3_4b.pt`

### 9.3 Entraînement SAE Qwen3-4B ✅
- [x] Pilot 5M tokens (quick test) : MSE=363, explained variance=99.97%, L0=17076/20480 (83%)
- [x] Config : d_in=2560, d_sae=20480 (8×), layer 18, L1=5e-3
- [x] wandb : https://wandb.ai/arthur-edmond-perso/sae-qwen3-4b/runs/ug6y3exq
- [ ] Full 20M tokens (à relancer si résultats prometteurs)

**Fichier :** `results/sae_qwen3_4b_L18_8x/`

### 9.4 Analyse domaine SAE 4B ✅
- [x] `python -m src.steering.analyze_sae_features --model Qwen/Qwen3-4B`

**Comparaison 0.6B vs 4B :**
| Métrique | Qwen3-0.6B | Qwen3-4B |
|----------|:----------:|:--------:|
| SAE features | 8,192 | 20,480 |
| Contrastive diffusion (math) | 57% | **45%** |
| Contrastive diffusion (law) | 59% | **51%** |
| Contrastive diffusion (history) | 47% | **41%** |
| Overlap math | 2/20 | 1/20 |
| Overlap law | 0/20 | **5/20** ★ |
| Overlap history | 4/20 | 0/20 |

**Résultat clé :** Les vecteurs contrastifs sont **moins diffus sur 4B** (41-51% vs 47-59%). Law montre 5/20 overlap — première fois qu'un domaine aligne significativement les vecteurs contrastifs avec les features SAE.

### 9.5 Feature-targeted benchmark 4B ✅
- [x] Law (n=50) : baseline 22%, **single α=30 24%** (+2) ★
- [x] Math (n=50) : baseline 48%, **single α=10 44%** (-4, meilleure préservation)
- [x] History (n=50) : baseline 34%, contrastive résistant jusqu'à α=30

**Résultats complets 4B (n=50, best per method) :**
| Domaine | Baseline | Contrastive best | Feature single best | Delta single vs contrastive |
|---------|:--------:|:----------------:|:-------------------:|:--------------------------:|
| Math | **48.0%** | 44.0% (α=10) | 44.0% (α=10) | 0pp (equal) |
| Law | **22.0%** | 24.0% (α=10) | 24.0% (α=30) | +8pp at α=30 ★ |
| History | **34.0%** | 34.0% (α=10-30) | 32.0% (α=10) | -2pp |

**Comparaison baselines 0.6B vs 4B :**
| Domaine | 0.6B | 4B | Gain |
|---------|:----:|:--:|:----:|
| Math | 18.0% | **48.0%** | +30pp |
| Law | 24.0% | 22.0% | -2pp |
| History | 14.0% | **34.0%** | +20pp |

**Conclusions :**
- Le 4B a des baselines beaucoup plus élevées (math +30pp, history +20pp)
- **Aucune méthode ne dépasse la baseline** de manière significative — ni sur 0.6B ni sur 4B
- Single-feature = stratégie la plus robuste : **préserve la baseline mieux** que contrastive à α modéré
- History sur 4B est remarquablement résistant au steering (34% maintenu jusqu'à α=30)
- **Verdict : le scaling ne résout pas le problème fondamental** — steering ≠ knowledge injection

### 9.6 Streamlit multi-modèle ✅
- [x] Sélecteur de modèle (0.6B / 4B) dans la sidebar
- [x] Chargement dynamique SAE, vecteurs contrastifs, et hook layer par modèle

### 9.7 Article & synthèse comparative ⬜
- [ ] Table comparative 0.6B vs 4B : SAE metrics, overlap, feature-targeted accuracy
- [ ] Mise à jour article/main.tex

---

## Phase 10 — LFM2-700M Full Pipeline ✅
> Reproduire l'intégralité du pipeline SAE + steering sur LFM2-700M (hybrid conv+attention) pour valider la généralité cross-architecture.

### 10.1 Vecteurs contrastifs LFM2-700M ✅
- [x] `python -m src.steering.mmlu_pro_vectors --model lfm2_700m`
- [x] 14 domaines × 16 layers × 1536d
- [x] Fichier : `results/mmlu_pro_vectors_lfm2_700m.pt`

### 10.2 SAE Training (custom HF hooks) ✅
- [x] TransformerLens ne supporte pas LFM2 → nouveau script `train_sae_hf.py`
- [x] Architecture-agnostique via `register_forward_hook` (bypass TransformerLens)
- [x] Config : d_in=1536, d_sae=12288 (8×), layer 8, 5M tokens
- [x] Final : MSE=0.062, explained variance=98.0%, alive=88%, L0=3190
- [x] wandb : projet `sae-lfm2_700m`
- [x] Fichier : `results/sae_lfm2_700m_L8_8x/`

### 10.3 Analyse domaine SAE ✅
- [x] `python -m src.steering.analyze_sae_features --model LiquidAI/LFM2-700M`

**Comparaison 0.6B vs 4B vs LFM2-700M :**
| Métrique | Qwen3-0.6B | Qwen3-4B | LFM2-700M |
|----------|:----------:|:--------:|:---------:|
| SAE features | 8,192 | 20,480 | 12,288 |
| Diffusion math | 57% | 45% | **38%** |
| Diffusion law | 59% | 51% | **28%** |
| Diffusion history | 47% | 41% | **26%** |
| Overlap math | 2/20 | 1/20 | 1/20 |
| Overlap law | 0/20 | 5/20 | 2/20 |
| Overlap history | 4/20 | 0/20 | 1/20 |

**Résultat clé :** L'architecture hybride conv+attention produit les vecteurs contrastifs **les moins diffus** (26-38% vs 47-59% pour Qwen 0.6B). Overlap reste minime.

### 10.4 Feature-targeted benchmark ✅
- [x] `python -m src.steering.feature_targeted_steering --model LiquidAI/LFM2-700M --limit 50`

| Domaine | Baseline | Contrastive best | Feature best | Delta |
|---------|:--------:|:----------------:|:------------:|:-----:|
| Math | **24.0%** | 8.0% (α=10) | 14.0% (α=10) | -10pp |
| Law | 14.0% | **22.0%** (α=10) ★ | **18.0%** (α=10) ★ | +8pp / +4pp |
| History | **22.0%** | 18.0% (α=10) | 18.0% (α=60) | -4pp |

**Conclusions LFM2 :**
- Law seul domaine positif (+8pp contrastive), même pattern que Qwen
- Math sévèrement dégradé (-16pp contrastive)
- Window effective plus étroite (α=60 dégrade law, contrairement à Qwen)
- **Architecture hybride ne change pas le constat : steering ≠ knowledge injection**

### 10.5 Steering Arena Web App ✅
- [x] React + FastAPI avec SSE streaming séquentiel
- [x] 3 modèles : Qwen3-0.6B, Qwen3-4B, LFM2-700M
- [x] Dynamic layer slider + SAE cross-layer warning
- [x] Vector space visualizations (PCA, cosine heatmap, norms)
- [x] KaTeX rendering pour les outputs mathématiques
- [x] Dockerfile multi-stage + justfile
- [x] Developed by Arthur EDMOND

### 10.6 Output-Score Selection + Multiplicative Scaling ✅
- [x] Implémentation `output_score.py` : sélection par projection W_dec × W_unembed
- [x] Hook multiplicatif : `h' = h × (1 + αv̂)` dans steering.py et feature_targeted_steering.py
- [x] Benchmark complet n=200, 3 domaines, 28 configs par domaine (84 evals total)
- [x] Output-score vectors générés pour Qwen3-0.6B

**Résultats n=200 Qwen3-0.6B (best per category) :**
| Domaine | Baseline | Best classique | Best output-score |
|---------|:--------:|:--------------:|:-----------------:|
| Math | 12.5% ±2.3 | 13.5% (contr α=60) | **13.5%** (outscore_single_mult α=10) |
| Law | **18.0%** ±2.7 | 18.0% (feat_uni α=60) | 17.0% (outscore_single_mult α=10) |
| History | 16.0% ±2.6 | 16.0% (contr_mult α=30) | **17.5%** (outscore_weighted α=10) ★ |

**Conclusions :**
- Output-score features = entièrement disjoints des input-diff features (0/20 overlap)
- α=10 est le seul coefficient qui fonctionne — α≥30 dégrade systématiquement
- History seul domaine avec signal positif (+1.5pp) mais dans la marge d'erreur
- **Amélioration méthodologique (moins de dégradation) mais pas de knowledge injection**
- Confirme le finding n=50→n=200 : les gains apparents à petit n sont des faux positifs

### 10.7 Article arXiv mis à jour ✅
- [x] Section output-score + multiplicative scaling (Table 8)
- [x] 14e contribution dans l'abstract
- [x] 2 nouvelles références (Acharya 2025, Stoehr 2024)
- [x] Conclusion mise à jour
- [x] 29 pages, compile clean, arxiv-submission.tar.gz prêt

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
├── lm-eval              # EleutherAI standardized evaluation (GSM8K, MMLU-Pro)
├── sae-lens             # SAE training & analysis
├── transformer-lens     # Hook-based model internals (SAE activation extraction)
├── wandb                # Training visualization
├── streamlit            # Interactive steering demo app (legacy)
├── fastapi + uvicorn    # Steering Arena API (SSE streaming)
├── react + vite + ts    # Steering Arena frontend (Bittensor DA)
├── katex                # LaTeX math rendering in frontend
└── docker               # Multi-stage containerization
```

## Modèles

| Modèle | Params | Architecture | Usage |
|--------|--------|-------------|-------|
| Qwen3-4B-Instruct-2507 | 4B | Transformer | Modèle principal, steering instruct |
| Qwen3-4B | 4B | Transformer | Steering base model (comparaison) |
| Qwen3-0.6B | 0.6B | Transformer (28L) | SLM instruct : GSM8K, SWE-bench, MMLU-Pro |
| Qwen3-0.6B-Base | 0.6B | Transformer (28L) | SLM base : GSM8K steering |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | Hybrid SSM+Attention (16L) | MMLU-Pro cross-architecture steering |
| LiquidAI/LFM2-700M | 0.7B | Hybrid Conv+GQA (16L) | Full SAE pipeline + MMLU-Pro benchmark |
| Llama-3.2-3B-Instruct | 3B | Transformer (32L) | Tokenizer + MMLU-Pro scaling test |
| Gemma-3-1B-IT | 1B | Transformer | Comparaison tokenizer |
| Phi-3-mini-4k-instruct | 3.8B | Transformer | Comparaison tokenizer |
