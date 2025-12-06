import argparse
import os

import esm
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from protein_config import Protein


def plot_head_attention(
    model,
    alphabet,
    sequence,
    layer,
    head,
    contact_pair=None,
    device="cpu",
    save_path=None,
):
    """
    Plots the attention heatmap for a specific Layer/Head.
    Highlights the contact_pair (i, j) if provided.
    Handles variable tensor shapes from different ESM versions.
    """
    print(f"Visualizing Layer {layer} Head {head}...")

    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        # Get attention weights
        results = model(batch_tokens, repr_layers=[], need_head_weights=True)
        attn = results["attentions"]  # Raw tensor

        # --- SHAPE FIX LOGIC ---
        # Different versions of ESM/PyTorch return different shapes.
        # We need to find which dimension matches the number of layers (6).

        # Expected shapes:
        # Type A: (Layers, Batch, Heads, Seq, Seq) -> Standard
        # Type B: (Batch, Layers, Heads, Seq, Seq) -> Common in some setups

        num_layers = len(model.layers)

        if attn.shape[0] == num_layers:
            # Case A: Layers are dim 0
            attn_matrix = attn[layer, 0, head].cpu().numpy()
        elif attn.shape[1] == num_layers:
            # Case B: Layers are dim 1 (Your previous error case)
            attn_matrix = attn[0, layer, head].cpu().numpy()
        else:
            print(
                f"Debug Info: Tensor shape is {attn.shape}, Num Layers is {num_layers}"
            )
            raise ValueError("Could not determine layer dimension in attention tensor.")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(attn_matrix, cmap="viridis", square=True)

    plt.title(f"Attention Map: Layer {layer} Head {head}")
    plt.xlabel("Key Position (Source)")
    plt.ylabel("Query Position (Destination)")

    # Highlight the contact point if provided
    if contact_pair:
        i, j = contact_pair
        # Note: Heatmap uses (col, row) for patches, which is (j, i)
        # We draw boxes at both (i,j) and (j,i) because attention is often symmetric-ish
        rect1 = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3)
        rect2 = plt.Rectangle((i, j), 1, 1, fill=False, edgecolor="red", lw=3)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        print(f"Highlighted contact at indices {contact_pair}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1, help="Layer index (0-based)")
    parser.add_argument("--head", type=int, default=4, help="Head index (0-based)")
    parser.add_argument(
        "--protein",
        type=str,
        choices=[p.name for p in Protein],
        help="Specific protein to visualize (if not set, all proteins will be visualized)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory for visualizations",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which proteins to process
    proteins_to_process = [Protein[args.protein]] if args.protein else list(Protein)

    for protein in proteins_to_process:
        print(f"\nProcessing {protein.protein_name}...")

        # Use default contact pair for Ubiquitin, None for others
        target_contact = (1, 17) if protein == Protein.UBIQUITIN else None

        # Generate filename
        safe_name = protein.protein_name.replace(" ", "_").lower()
        save_path = os.path.join(
            args.output_dir, f"{safe_name}_L{args.layer}H{args.head}.png"
        )

        plot_head_attention(
            model,
            alphabet,
            protein.sequence,
            layer=args.layer,
            head=args.head,
            contact_pair=target_contact,
            device=device,
            save_path=save_path,
        )

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
