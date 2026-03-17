"""
Train a Sparse Autoencoder on a model's residual stream using raw HuggingFace hooks.

Bypasses TransformerLens — works with ANY model architecture (Transformer, SSM, hybrid).
Saves output in SAELens-compatible format so analyze_sae_features.py can load it.

Usage:
    python -m src.steering.train_sae_hf --model LiquidAI/LFM2-700M --layer 8 --d_in 1536
    python -m src.steering.train_sae_hf --model Qwen/Qwen3-0.6B --layer 14 --d_in 1024
"""

import argparse
import functools
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MODEL_PRESETS = {
    "Qwen/Qwen3-0.6B": (14, 1024),
    "Qwen/Qwen3-4B": (18, 2560),
    "LiquidAI/LFM2-700M": (8, 1536),
}


# ── Sparse Autoencoder ──────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """Standard SAE: encoder with ReLU, decoder with unit-norm columns."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Kaiming init
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Normalize decoder columns
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ── Activation collection ───────────────────────────────────────────────────

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    raise ValueError(f"Cannot find layers in {type(model)}")


class ActivationBuffer:
    """Collects activations from a model layer via forward hooks."""

    def __init__(self, model, tokenizer, layer_idx: int, context_size: int,
                 buffer_size: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.context_size = context_size
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self._hook_output = None

        # Dataset — stream OpenWebText
        self.dataset = iter(load_dataset(
            "Skylion007/openwebtext", split="train", streaming=True        ))

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._hook_output = hidden.detach()

    def fill_buffer(self):
        """Fill buffer with activations."""
        self.buffer = []
        layers = get_layers(self.model)
        handle = layers[self.layer_idx].register_forward_hook(self._hook_fn)

        tokens_collected = 0
        try:
            while tokens_collected < self.buffer_size:
                # Get next batch of text
                try:
                    text = next(self.dataset)["text"]
                except StopIteration:
                    self.dataset = iter(load_dataset(
                        "Skylion007/openwebtext", split="train", streaming=True                    ))
                    text = next(self.dataset)["text"]

                inputs = self.tokenizer(
                    text, return_tensors="pt", max_length=self.context_size,
                    truncation=True, padding=False
                ).to(self.device)

                if inputs["input_ids"].shape[1] < 4:
                    continue

                with torch.no_grad():
                    self.model(**inputs)

                # _hook_output: [1, seq_len, d_in]
                acts = self._hook_output.squeeze(0).float()  # [seq_len, d_in]
                self.buffer.append(acts.cpu())
                tokens_collected += acts.shape[0]
        finally:
            handle.remove()

        self.buffer = torch.cat(self.buffer, dim=0)[:self.buffer_size]
        return self.buffer


# ── Training loop ───────────────────────────────────────────────────────────

def train_sae(model, tokenizer, layer_idx: int, d_in: int, d_sae: int,
              training_tokens: int, device: str, l1_coeff: float, lr: float,
              context_size: int, batch_size: int = 4096, buffer_batches: int = 64,
              use_wandb: bool = False, wandb_project: str = "sae-hf"):

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "layer": layer_idx, "d_in": d_in, "d_sae": d_sae,
                    "training_tokens": training_tokens, "l1_coefficient": l1_coeff,
                    "lr": lr, "batch_size": batch_size, "context_size": context_size,
                },
            )
        except Exception as e:
            print(f"wandb init failed ({e}), continuing without logging")
            use_wandb = False

    sae = SparseAutoencoder(d_in, d_sae).to(device).float()
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    # Warmup
    warmup_steps = 500
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, step / warmup_steps)
    )

    buffer = ActivationBuffer(
        model, tokenizer, layer_idx, context_size,
        buffer_size=batch_size * buffer_batches, device=device,
    )

    tokens_seen = 0
    step = 0
    total_steps = training_tokens // batch_size
    t0 = time.time()

    print(f"\nTraining {total_steps} steps ({training_tokens:,} tokens, batch={batch_size})")
    print("-" * 60)

    while tokens_seen < training_tokens:
        # Refill buffer
        print(f"  Filling activation buffer ({batch_size * buffer_batches:,} tokens)...")
        acts = buffer.fill_buffer().to(device)

        # Shuffle
        perm = torch.randperm(acts.shape[0])
        acts = acts[perm]

        # Mini-batch training over buffer
        for i in range(0, acts.shape[0] - batch_size + 1, batch_size):
            batch = acts[i:i + batch_size]

            x_hat, z = sae(batch)

            # Reconstruction loss (MSE)
            mse_loss = (batch - x_hat).pow(2).sum(dim=-1).mean()

            # Sparsity loss (L1 on activations)
            l1_loss = z.abs().sum(dim=-1).mean()

            loss = mse_loss + l1_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Re-normalize decoder columns
            with torch.no_grad():
                norms = sae.W_dec.data.norm(dim=1, keepdim=True)
                sae.W_dec.data = sae.W_dec.data / (norms + 1e-8)

            tokens_seen += batch_size
            step += 1

            if step % 100 == 0:
                # Compute stats
                with torch.no_grad():
                    explained_var = 1.0 - (batch - x_hat).var() / (batch.var() + 1e-8)
                    alive = (z > 0).any(dim=0).float().mean()
                    l0 = (z > 0).float().sum(dim=-1).mean()

                elapsed = time.time() - t0
                tok_per_sec = tokens_seen / elapsed
                print(
                    f"  step {step:>5d}/{total_steps} | "
                    f"loss={loss.item():.4f} mse={mse_loss.item():.4f} l1={l1_loss.item():.4f} | "
                    f"var={explained_var.item():.4f} alive={alive.item():.2%} L0={l0.item():.0f} | "
                    f"{tok_per_sec:.0f} tok/s"
                )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "loss": loss.item(),
                        "mse_loss": mse_loss.item(),
                        "l1_loss": l1_loss.item(),
                        "explained_variance": explained_var.item(),
                        "alive_features": alive.item(),
                        "L0": l0.item(),
                        "tokens_seen": tokens_seen,
                        "tok_per_sec": tok_per_sec,
                        "lr": scheduler.get_last_lr()[0],
                    }, step=step)

            if tokens_seen >= training_tokens:
                break

    elapsed = time.time() - t0
    print(f"\nTraining complete: {tokens_seen:,} tokens in {elapsed:.0f}s ({tokens_seen/elapsed:.0f} tok/s)")

    if use_wandb:
        import wandb
        wandb.finish()

    return sae


# ── Save in SAELens-compatible format ───────────────────────────────────────

def save_sae(sae: SparseAutoencoder, out_path: Path, model_name: str,
             hook_name: str, hook_layer: int):
    out_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    state = {
        "W_enc": sae.W_enc.data.cpu(),
        "b_enc": sae.b_enc.data.cpu(),
        "W_dec": sae.W_dec.data.cpu(),
        "b_dec": sae.b_dec.data.cpu(),
    }
    torch.save(state, out_path / "sae_weights.pt")

    # Save config for SAELens compatibility
    cfg = {
        "d_in": sae.d_in,
        "d_sae": sae.d_sae,
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_layer": hook_layer,
        "dtype": "float32",
        "architecture": "standard",
    }
    with open(out_path / "cfg.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"SAE saved to: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SAE using HF hooks (any architecture)")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2-700M")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--d_in", type=int, default=None)
    parser.add_argument("--training_tokens", type=int, default=5_000_000)
    parser.add_argument("--expansion_factor", type=int, default=8)
    parser.add_argument("--l1_coefficient", type=float, default=5e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    preset = MODEL_PRESETS.get(args.model, (8, 1536))
    hook_layer = args.layer if args.layer is not None else preset[0]
    d_in = args.d_in if args.d_in is not None else preset[1]
    d_sae = d_in * args.expansion_factor
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print(f"SAE TRAINING (HF hooks) — {args.model} Layer {hook_layer}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_sae={d_sae} ({args.expansion_factor}x)")
    print(f"Training tokens: {args.training_tokens:,}")

    # Load model
    dtype = torch.bfloat16 if "LFM2" in args.model else torch.float16
    print(f"Loading model ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # Train
    wandb_project = args.wandb_project or f"sae-{model_short}"
    sae = train_sae(
        model, tokenizer, hook_layer, d_in, d_sae,
        args.training_tokens, device, args.l1_coefficient, args.lr,
        args.context_size, args.batch_size,
        use_wandb=args.wandb, wandb_project=wandb_project,
    )

    # Save
    out_path = RESULTS_DIR / f"sae_{model_short}_L{hook_layer}_{args.expansion_factor}x"
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    save_sae(sae, out_path, args.model, hook_name, hook_layer)

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
