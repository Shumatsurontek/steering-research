"""ModelManager singleton — one model in memory at a time."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from transformer_lens import HookedTransformer

logger = logging.getLogger("steering.manager")

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

TARGET_DOMAINS = ["math", "law", "history"]

DOMAIN_PROMPTS = {
    "math": [
        "Solve the equation 3x + 7 = 22 for x.",
        "What is the derivative of sin(x) * cos(x)?",
        "Prove that the square root of 2 is irrational.",
        "Calculate the integral of e^x from 0 to 1.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "What is the probability of rolling two sixes with two dice?",
        "Simplify the expression (x^2 - 4)/(x - 2).",
        "How many ways can you arrange 5 books on a shelf?",
        "What is the Taylor series expansion of ln(1+x)?",
        "Solve the differential equation dy/dx = 2xy.",
    ],
    "law": [
        "What is the difference between civil and criminal law?",
        "Explain the concept of habeas corpus.",
        "What are the elements of a valid contract?",
        "Define the legal principle of stare decisis.",
        "What is the Miranda warning and when must it be given?",
        "Explain the doctrine of sovereign immunity.",
        "What constitutes negligence in tort law?",
        "What is the difference between a felony and a misdemeanor?",
        "Explain the concept of due process under the 14th Amendment.",
        "What are the requirements for obtaining a patent?",
    ],
    "history": [
        "What caused the fall of the Roman Empire?",
        "Describe the main events of the French Revolution.",
        "What was the significance of the Magna Carta?",
        "Explain the causes of World War I.",
        "What was the impact of the Industrial Revolution on society?",
        "Describe the civil rights movement in the United States.",
        "What were the consequences of the Treaty of Versailles?",
        "Explain the rise and fall of the Ottoman Empire.",
        "What was the significance of the Silk Road?",
        "Describe the colonization of the Americas by European powers.",
    ],
}

MODEL_CONFIGS = {
    "Qwen3-0.6B": {
        "model_id": "Qwen/Qwen3-0.6B",
        "layer": 14,
        "sae_dir": "sae_qwen3_0.6b_L14_8x", # TODO: change to the correct SAE directory
        "vectors": "mmlu_pro_vectors_qwen3_0.6b.pt", # TODO: change to the correct vectors file
        "params": "0.6B",
        "layers_total": 28,
    },
    "Qwen3-4B": {
        "model_id": "Qwen/Qwen3-4B",
        "layer": 18,
        "sae_dir": "sae_qwen3_4b_L18_8x",
        "vectors": "mmlu_pro_vectors_qwen3_4b.pt",
        "params": "4B",
        "layers_total": 36,
    },
    "LFM2-700M": {
        "model_id": "LiquidAI/LFM2-700M",
        "layer": 8,
        "sae_dir": "sae_lfm2_700m_L8_8x",
        "vectors": "mmlu_pro_vectors_lfm2_700m.pt",
        "params": "0.7B",
        "layers_total": 16,
    },
}


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    """Manages a single model + vectors in memory with async locking."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.current_model_key: str | None = None
        self.model = None
        self.tokenizer = None
        self.contrastive_vectors: dict | None = None
        self.feature_vectors: dict | None = None
        self.feature_info: dict | None = None
        self.device = _get_device()

    async def load_model(self, model_key: str) -> None:
        async with self._lock:
            if self.current_model_key == model_key:
                logger.info("Model %s already loaded", model_key)
                return

            cfg = MODEL_CONFIGS[model_key]
            logger.info("Loading model %s on %s...", cfg["model_id"], self.device)

            # Unload previous
            self._unload()

            loop = asyncio.get_event_loop()
            self.model, self.tokenizer = await loop.run_in_executor(
                None, self._load_hf_model, cfg["model_id"]
            )
            self.current_model_key = model_key

            # Load contrastive vectors
            vec_path = RESULTS_DIR / cfg["vectors"]
            if vec_path.exists():
                self.contrastive_vectors = await loop.run_in_executor(
                    None, lambda: torch.load(vec_path, map_location="cpu", weights_only=True)
                )
                logger.info("Contrastive vectors loaded from %s", vec_path)

            # Load SAE feature vectors
            sae_path = RESULTS_DIR / cfg["sae_dir"]
            if sae_path.exists():
                self.feature_vectors, self.feature_info = await loop.run_in_executor(
                    None, self._build_feature_vectors, cfg["model_id"], str(sae_path), cfg["layer"], 20
                )
                logger.info("SAE feature vectors built")

    def _load_hf_model(self, model_id: str):
        t0 = time.time()
        logger.info("Loading tokenizer for %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Loading model weights for %s → %s", model_id, self.device)
        # Use bfloat16 for LFM2 (native), float16 for others
        dtype = torch.bfloat16 if "LFM2" in model_id else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        model.eval()
        logger.info("Model loaded in %.1fs", time.time() - t0)
        return model, tokenizer

    def _load_sae_weights(self, sae_path: str):
        """Load SAE — try SAELens format first, then custom format."""
        p = Path(sae_path)
        # Custom format (train_sae_hf.py)
        weights_file = p / "sae_weights.pt"
        if weights_file.exists():
            logger.info("Loading custom SAE from %s", weights_file)
            state = torch.load(weights_file, map_location=self.device, weights_only=True)
            return state["W_enc"], state["b_enc"], state["W_dec"], state["b_dec"]
        # SAELens format
        logger.info("Loading SAELens SAE from %s", sae_path)
        sae = SAE.load_from_disk(sae_path, device=self.device)
        return sae.W_enc.detach(), sae.b_enc.detach(), sae.W_dec.detach(), sae.b_dec.detach()

    def _encode_with_sae(self, x, W_enc, b_enc):
        """SAE encode: ReLU(x @ W_enc + b_enc)."""
        return torch.relu(x @ W_enc.to(x.device, dtype=x.dtype) + b_enc.to(x.device, dtype=x.dtype))

    def _build_feature_vectors(self, model_id: str, sae_path: str, layer: int, top_k: int):
        t0 = time.time()
        logger.info("Building SAE feature vectors (layer=%d, top_k=%d)", layer, top_k)

        W_enc, b_enc, W_dec_raw, b_dec = self._load_sae_weights(sae_path)

        # Try TransformerLens path first (Qwen, Llama, etc.)
        try:
            hook_name = f"blocks.{layer}.hook_resid_post"
            tl_model = HookedTransformer.from_pretrained_no_processing(model_id, device=self.device)
            activations = {}
            for domain, prompts in DOMAIN_PROMPTS.items():
                all_acts = []
                for prompt in prompts:
                    tokens = tl_model.to_tokens(prompt)
                    _, cache = tl_model.run_with_cache(tokens, stop_at_layer=layer + 1)
                    residual = cache[hook_name].squeeze(0).float()
                    feat_acts = self._encode_with_sae(residual, W_enc, b_enc)
                    all_acts.append(feat_acts.mean(dim=0).detach().cpu())
                activations[domain] = torch.stack(all_acts).mean(dim=0)
            del tl_model
        except (ValueError, Exception) as e:
            # Fallback: HuggingFace hooks (LFM2, other unsupported archs)
            logger.info("TransformerLens unavailable for %s, using HF hooks: %s", model_id, e)
            activations = self._collect_activations_hf(model_id, layer, W_enc, b_enc)

        domains = list(activations.keys())
        W_dec = W_dec_raw.detach().cpu().float()

        feature_vectors = {}
        feature_info = {}
        for domain in TARGET_DOMAINS:
            domain_mean = activations[domain]
            other_mean = torch.stack(
                [activations[d] for d in domains if d != domain]
            ).mean(dim=0)
            differential = domain_mean - other_mean

            topk = differential.topk(top_k)
            top_indices = topk.indices
            top_values = topk.values

            weighted_vec = torch.zeros(W_dec.shape[1])
            for idx, val in zip(top_indices, top_values):
                weighted_vec += val.item() * W_dec[idx]
            uniform_vec = W_dec[top_indices].sum(dim=0)
            single_vec = W_dec[top_indices[0]].clone()

            feature_vectors[domain] = {
                "weighted": weighted_vec,
                "uniform": uniform_vec,
                "single": single_vec,
            }
            feature_info[domain] = {
                "top_features": top_indices[:10].tolist(),
                "top_diffs": [f"{v:.3f}" for v in top_values[:10].tolist()],
            }

        del W_enc, b_enc, W_dec_raw, b_dec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SAE feature vectors built in %.1fs (%d domains)", time.time() - t0, len(TARGET_DOMAINS))
        return feature_vectors, feature_info

    def _collect_activations_hf(self, model_id, layer, W_enc, b_enc):
        """Collect SAE activations using HuggingFace model + forward hooks."""
        import functools
        dtype = torch.bfloat16 if "LFM2" in model_id else torch.float16
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        hf_model.eval()
        hf_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        def get_layers_hf(m):
            if hasattr(m, "model") and hasattr(m.model, "layers"):
                return m.model.layers
            if hasattr(m, "model") and hasattr(m.model, "blocks"):
                return m.model.blocks
            raise ValueError(f"Cannot find layers in {type(m)}")

        layers_list = get_layers_hf(hf_model)
        hook_output = {}

        def hook_fn(module, input, output, key="act"):
            hidden = output[0] if isinstance(output, tuple) else output
            hook_output[key] = hidden.detach()

        activations = {}
        for domain, prompts in DOMAIN_PROMPTS.items():
            all_acts = []
            for prompt in prompts:
                handle = layers_list[layer].register_forward_hook(
                    functools.partial(hook_fn, key="act")
                )
                inputs = hf_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    hf_model(**inputs)
                handle.remove()
                residual = hook_output["act"].squeeze(0).float()
                feat_acts = self._encode_with_sae(residual, W_enc, b_enc)
                all_acts.append(feat_acts.mean(dim=0).detach().cpu())
            activations[domain] = torch.stack(all_acts).mean(dim=0)

        del hf_model, hf_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return activations

    async def recalculate_sae(self, model_key: str, top_k: int = 20) -> None:
        async with self._lock:
            cfg = MODEL_CONFIGS[model_key]
            sae_path = RESULTS_DIR / cfg["sae_dir"]
            if not sae_path.exists():
                raise FileNotFoundError(f"SAE directory not found: {sae_path}")

            loop = asyncio.get_event_loop()
            self.feature_vectors, self.feature_info = await loop.run_in_executor(
                None, self._build_feature_vectors, cfg["model_id"], str(sae_path), cfg["layer"], top_k
            )

    def _unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.contrastive_vectors = None
        self.feature_vectors = None
        self.feature_info = None
        self.current_model_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status(self) -> dict:
        return {
            "loaded_model": self.current_model_key,
            "has_contrastive": self.contrastive_vectors is not None,
            "has_sae": self.feature_vectors is not None,
        }
