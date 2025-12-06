import argparse
import sys

import torch
from esm import pretrained

MODEL_CHOICES = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


def load_model(model_name: str, device: torch.device):
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {MODEL_CHOICES}")
    loader = getattr(pretrained, model_name)
    model, alphabet = loader()
    model = model.eval().to(device)
    return model, alphabet


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Load an ESM-2 model and print hook-relevant info."
    )
    parser.add_argument(
        "--model",
        default="esm2_t12_35M_UR50D",
        choices=MODEL_CHOICES,
        help="Which ESM-2 checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load the model on.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, alphabet = load_model(args.model, device)

    num_layers = len(model.layers)
    head_counts = [blk.self_attn.num_heads for blk in model.layers]
    unique_heads = sorted(set(head_counts))
    dim = (
        model.embed_dim
        if hasattr(model, "embed_dim")
        else model.layers[0].self_attn.embed_dim
    )

    print(f"Loaded {args.model} on {device}")
    print(f"Vocabulary size: {len(alphabet)}")
    print(f"Layers: {num_layers}")
    print(f"Embedding dim: {dim}")
    if len(unique_heads) == 1:
        print(f"Heads per layer: {unique_heads[0]}")
    else:
        print(f"Heads per layer (varying): {head_counts}")

    has_mlp = hasattr(model.layers[0], "mlp")
    has_fc2 = hasattr(model.layers[0], "fc2")
    print("Hook targets:")
    print(" - Attention: layer.self_attn (per-head outputs)")
    if has_mlp:
        print(" - MLP: layer.mlp")
    elif has_fc2:
        print(" - MLP/FFN second proj: layer.fc2")
    else:
        print(" - MLP: not found on layer blocks")


if __name__ == "__main__":
    main(sys.argv[1:])
