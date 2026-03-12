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

### 2.1 Exploration via Neuronpedia
- [ ] Utiliser le circuit tracer Qwen3-4B (TRANSCODER-HP MLP 164k) — *future work*

### 2.2 SAE-inspired contrastive analysis ✅
- [x] Hook-based activation extraction aux 36 couches de Qwen3-4B
- [x] 10 prompts calendrier vs 10 prompts neutres (contrastive pairs)
- [x] Vecteurs de steering = mean difference par couche (last-token position)
- [x] Layer importance ranking par L2 norm et cosine distance
- [x] Logit lens sur les top-3 couches (layers 33-35)

**Résultats clés :**
- Couche 35 = plus discriminative (L2 norm: 361.0, cosine dist: 0.151)
- Logit lens confirme : `schedule`, `agenda`, `attendees`, `RSVP`, `calendar` promus
- Signal cross-lingue détecté (tokens chinois email/modify promus)

---

## Phase 3 — Steering Vectors ✅
> Construire et tester des vecteurs de steering pour le task calendrier

### 3.1 Construction et application des vecteurs ✅
- [x] Méthode contrastive : 10 paires calendrier/neutre
- [x] Vecteurs de différence par couche sauvegardés (`steering_vectors.pt`)
- [x] Sweep de coefficients (α = 0, 5, 15, 30, 50) à la couche 35
- [x] Test sur 6 prompts (calendrier, ambigus, non-calendrier)

**Résultat clé : le steering est INEFFICACE sur modèles instruct.**
Réponses identiques à tous les coefficients. L'instruction tuning sature le comportement.

### 3.2 Méthode budget guidance (inspiré arxiv:2506.13752)
- [ ] Implémenter le contrôle de longueur de raisonnement — *future work*

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

## Phase 5 — Intégration Agentique
> Tester en conditions réelles avec Ollama/llama.cpp + LangChain — *future work*

- [ ] Servir Qwen3-4B via Ollama
- [ ] Agent LangChain avec tool "create_calendar_event"
- [ ] Évaluation comparative sur les 29 cas avec les 5 stratégies
- [ ] Mid-layer steering (couches 15-25) vs last-layer
- [ ] Steering sur modèle base (non-instruct) pour comparaison

---

## Phase 6 — Rédaction de l'Article ✅
> Synthèse des résultats

- [x] Article LaTeX 8 pages, format arxiv-ready
- [x] Sections : Introduction, Related Work, Tokenizer Analysis, Feature Extraction, Steering, Baselines, Discussion
- [x] Compilé en PDF (`article/main.pdf`)

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
└── huggingface-hub 1.6.0 # Model downloads
```

## Modèles

| Modèle | Params | Usage |
|--------|--------|-------|
| Qwen3-4B-Instruct-2507 | 4B | Modèle principal, steering + agent |
| Gemma-3-1B-IT | 1B | Comparaison tokenizer |
| Llama-3.2-3B-Instruct | 3B | Comparaison tokenizer |
| Phi-3-mini-4k-instruct | 3.8B | Comparaison tokenizer |
