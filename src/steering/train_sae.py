"""
Train a Sparse Autoencoder on a model's residual stream.

Uses SAELens to train on the mid-layer sweet spot identified in our
layer×coefficient sweep. The resulting features can be compared to
our contrastive steering vectors.

Usage:
    python -m src.steering.train_sae
    python -m src.steering.train_sae --model Qwen/Qwen3-4B --layer 18 --d_in 2560
    python -m src.steering.train_sae --training_tokens 20_000_000 --l1_coefficient 0.05
"""

import argparse
from pathlib import Path

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)
from sae_lens.saes.sae import SAEMetadata

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CHECKPOINT_DIR = RESULTS_DIR / "sae_checkpoints"

# Model presets: (model_name, hook_layer, d_in)
MODEL_PRESETS = {
    "Qwen/Qwen3-0.6B": (14, 1024),
    "Qwen/Qwen3-4B": (18, 2560),
}


def main():
    parser = argparse.ArgumentParser(description="Train SAE on model residual stream")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Hook layer (default: from preset or 14)")
    parser.add_argument("--d_in", type=int, default=None,
                        help="Hidden dimension (default: from preset or 1024)")
    parser.add_argument("--training_tokens", type=int, default=5_000_000,
                        help="Total training tokens (default: 5M)")
    parser.add_argument("--expansion_factor", type=int, default=8,
                        help="SAE expansion factor (default: 8)")
    parser.add_argument("--l1_coefficient", type=float, default=5e-3,
                        help="L1 sparsity penalty (default: 5e-3)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--context_size", type=int, default=128,
                        help="Context window for activations (default: 128)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (default: sae-{model_short})")
    args = parser.parse_args()

    # Resolve model preset
    preset = MODEL_PRESETS.get(args.model, (14, 1024))
    hook_layer = args.layer if args.layer is not None else preset[0]
    d_in = args.d_in if args.d_in is not None else preset[1]
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    hook_name = f"blocks.{hook_layer}.hook_resid_post"
    wandb_project = args.wandb_project or f"sae-{model_short}"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    d_sae = d_in * args.expansion_factor

    print("=" * 60)
    print(f"SAE TRAINING — {args.model} Layer {hook_layer}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hook: {hook_name}")
    print(f"d_in={d_in}, d_sae={d_sae} ({args.expansion_factor}x)")
    print(f"Training tokens: {args.training_tokens:,}")
    print(f"L1 coefficient: {args.l1_coefficient}")
    print(f"Learning rate: {args.lr}")
    print(f"Context size: {args.context_size}")

    # SAE config (passed as the 'sae' parameter)
    sae_cfg = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device=device,
        l1_coefficient=args.l1_coefficient,
        metadata=SAEMetadata(
            model_name=args.model,
            hook_name=hook_name,
            hook_layer=hook_layer,
        ),
    )

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,

        # Model
        model_name=args.model,
        model_class_name="HookedTransformer",
        hook_name=hook_name,

        # Dataset — use untokenized openwebtext, streamed
        dataset_path="Skylion007/openwebtext",
        is_dataset_tokenized=False,
        streaming=True,
        context_size=args.context_size,

        # Training
        training_tokens=args.training_tokens,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=32,
        n_batches_in_buffer=64,
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=500,

        # Sparsity
        dead_feature_window=1000,
        dead_feature_threshold=1e-8,
        feature_sampling_window=2000,

        # Checkpoints
        n_checkpoints=2,
        checkpoint_path=str(CHECKPOINT_DIR),

        # Device
        device=device,
        dtype="float32",

        # Logging
        logger=LoggingConfig(
            log_to_wandb=args.wandb,
            wandb_project=wandb_project if args.wandb else None,
            wandb_log_frequency=10,
        ),
    )

    print(f"\nStarting training...")
    runner = LanguageModelSAETrainingRunner(cfg)
    sae = runner.run()

    # Save the trained SAE
    out_path = RESULTS_DIR / f"sae_{model_short}_L{hook_layer}_{args.expansion_factor}x"
    sae.save_model(str(out_path))
    print(f"\nSAE saved to: {out_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
