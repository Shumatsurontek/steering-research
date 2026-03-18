"""
Output-score feature selection for SAE steering vectors.

Based on: "SAEs Are Good for Steering -- If You Select the Right Features" (arXiv:2505.20063)

Instead of selecting features by input activation (contrastive mean-diff),
rank features by their OUTPUT INFLUENCE: how much each SAE decoder column,
projected through the unembedding matrix, promotes domain-relevant tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_unembedding(model) -> torch.Tensor:
    """Extract the unembedding (lm_head) matrix from a HuggingFace model."""
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()  # (vocab_size, hidden_dim)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        # Weight-tied models
        return model.model.embed_tokens.weight.detach()
    raise ValueError(f"Cannot find unembedding matrix in {type(model)}")


def compute_domain_token_distribution(
    tokenizer, domain_prompts: list[str], neutral_prompts: list[str],
    model, device: str = "cpu",
) -> torch.Tensor:
    """
    Compute a contrastive token distribution: which tokens are MORE likely
    in domain outputs vs neutral outputs.

    Returns: (vocab_size,) tensor of differential log-probabilities.
    """
    def avg_logits(prompts):
        all_logits = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            # Take logits at last token position
            last_logits = out.logits[0, -1, :].float().cpu()
            all_logits.append(last_logits)
        return torch.stack(all_logits).mean(dim=0)

    domain_logits = avg_logits(domain_prompts)
    neutral_logits = avg_logits(neutral_prompts)

    # Differential: which tokens are more promoted by domain prompts
    diff = F.log_softmax(domain_logits, dim=-1) - F.log_softmax(neutral_logits, dim=-1)
    return diff


def compute_output_scores(
    W_dec: torch.Tensor,
    W_unembed: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    """
    For each SAE feature j, compute how much its decoder column aligns
    with the target token distribution when projected through unembedding.

    Args:
        W_dec: (d_sae, d_in) SAE decoder weights
        W_unembed: (vocab_size, d_in) unembedding matrix
        target_distribution: (vocab_size,) differential log-prob distribution

    Returns:
        (d_sae,) output scores per feature
    """
    # Project each decoder column through unembedding: (d_sae, vocab_size)
    # All on CPU — this is a one-time computation
    projected = W_dec.float().cpu() @ W_unembed.float().cpu().T  # (d_sae, vocab_size)

    # Normalize both to unit vectors for cosine similarity
    proj_norm = F.normalize(projected, dim=-1)
    target_norm = F.normalize(target_distribution.float().cpu().unsqueeze(0), dim=-1)

    # Cosine similarity between each feature's output effect and the target
    scores = (proj_norm * target_norm).sum(dim=-1)  # (d_sae,)

    return scores


def build_output_scored_vectors(
    model_id: str,
    sae_W_dec: torch.Tensor,
    domain_prompts: dict[str, list[str]],
    neutral_prompts: list[str],
    top_k: int = 20,
    device: str = "cpu",
) -> dict:
    """
    Build steering vectors using output-score feature selection.

    Returns dict[domain] with keys:
        - "output_weighted": weighted sum of top-k output-scored decoder columns
        - "output_uniform": uniform sum of top-k output-scored decoder columns
        - "output_single": single best output-scored decoder column
        - "output_scores": raw scores for analysis
        - "top_features": indices of selected features
    """
    import gc

    dtype = torch.bfloat16 if "LFM2" in model_id else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    W_unembed = get_unembedding(model)
    print(f"    W_unembed: {W_unembed.shape}")

    vectors = {}
    for domain, prompts in domain_prompts.items():
        print(f"    [{domain}] Computing output scores...")
        target_dist = compute_domain_token_distribution(
            tokenizer, prompts, neutral_prompts, model, device
        )

        scores = compute_output_scores(sae_W_dec, W_unembed, target_dist)

        # Select top-k by output score
        topk = scores.topk(top_k)
        top_indices = topk.indices
        top_values = topk.values

        W_dec_cpu = sae_W_dec.float().cpu()

        weighted_vec = torch.zeros(W_dec_cpu.shape[1])
        for idx, val in zip(top_indices, top_values):
            weighted_vec += val.item() * W_dec_cpu[idx]
        uniform_vec = W_dec_cpu[top_indices].sum(dim=0)
        single_vec = W_dec_cpu[top_indices[0]].clone()

        vectors[domain] = {
            "output_weighted": weighted_vec,
            "output_uniform": uniform_vec,
            "output_single": single_vec,
            "output_scores": scores.cpu(),
            "top_features": top_indices.tolist(),
            "top_scores": [f"{v:.4f}" for v in top_values.tolist()],
        }

        print(f"      Top-5 features: {top_indices[:5].tolist()}")
        print(f"      Top-5 scores: {[f'{v:.4f}' for v in top_values[:5].tolist()]}")
        print(f"      Weighted norm: {weighted_vec.norm():.4f}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectors


# Neutral prompts for contrastive token distribution
NEUTRAL_PROMPTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Python is a high-level programming language known for its readability.",
    "The solar system has eight planets orbiting the Sun.",
    "DNA stands for deoxyribonucleic acid, which carries genetic information.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "The Earth orbits the Sun once every 365.25 days.",
]
