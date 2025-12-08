import argparse
import json
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
    protein_name=None,
    is_corrupted=False,
):
    """
    Plots the attention heatmap for a specific Layer/Head.
    Highlights the contact_pair (i, j) if provided.
    Handles variable tensor shapes from different ESM versions.

    Args:
        is_corrupted: Whether this is a corrupted sequence (for title annotation)
        protein_name: Name of protein (for title)
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

        num_layers = len(model.layers)

        if attn.shape[0] == num_layers:
            # Case A: Layers are dim 0
            attn_matrix = attn[layer, 0, head].cpu().numpy()
        elif attn.shape[1] == num_layers:
            # Case B: Layers are dim 1
            attn_matrix = attn[0, layer, head].cpu().numpy()
        else:
            print(
                f"Debug Info: Tensor shape is {attn.shape}, Num Layers is {num_layers}"
            )
            raise ValueError("Could not determine layer dimension in attention tensor.")

    # Plotting
    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(attn_matrix, cmap="viridis", square=True)

    # Build title
    seq_type = " [CORRUPTED]" if is_corrupted else ""
    protein_prefix = f"{protein_name}{seq_type} - " if protein_name else ""
    title = f"{protein_prefix}Layer {layer} Head {head}"
    if contact_pair:
        title += f"\nContact: {contact_pair}"

    plt.title(title)
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
        help="Specific protein to visualize (if not set, all proteins will be visualized)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--use-corrupted",
        action="store_true",
        help="Use corrupted sequences from experimental metadata (requires results/experimental_metadata.json)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load experimental metadata if available (for contact pairs and optional corrupted sequences)
    experimental_metadata = None
    metadata_path = "results/experimental_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            experimental_metadata = json.load(f)
        print(f"Loaded experimental metadata for {len(experimental_metadata)} proteins")
        print("Using contact pairs from experimental metadata")
    elif args.use_corrupted:
        print(f"ERROR: {metadata_path} not found!")
        print(
            "Please run multi_protein_main.py first to generate experimental metadata."
        )
        return
    else:
        print(
            f"WARNING: {metadata_path} not found. Contact pairs will not be highlighted."
        )
        print(
            "Run multi_protein_main.py first to generate experimental metadata with contact pairs."
        )

    # Determine which proteins to process
    if args.protein:
        # Single protein specified
        if args.use_corrupted and args.protein not in experimental_metadata:
            print(f"ERROR: No metadata found for protein '{args.protein}'")
            print(f"Available proteins: {list(experimental_metadata.keys())}")
            return
        proteins_to_process = [args.protein]
    else:
        # All proteins
        if args.use_corrupted:
            proteins_to_process = list(experimental_metadata.keys())
        else:
            proteins_to_process = [p.protein_name for p in Protein]

    for protein_name in proteins_to_process:
        print(f"\nProcessing {protein_name}...")

        # Get contact pair from metadata if available
        contact_pair = None
        if experimental_metadata and protein_name in experimental_metadata:
            contact_pair = tuple(experimental_metadata[protein_name]["contact_pair"])

        if args.use_corrupted:
            # Use corrupted sequence from metadata
            metadata = experimental_metadata[protein_name]
            sequence = metadata["corrupted_sequence"]
            is_corrupted = True
            print(f"  Using CORRUPTED sequence")
            print(f"  Contact: {contact_pair}")
            print(f"  Corruption method: {metadata['corruption_method']}")
        else:
            # Use clean sequence
            protein = next(p for p in Protein if p.protein_name == protein_name)
            sequence = protein.sequence
            is_corrupted = False
            print(f"  Using CLEAN sequence")
            if contact_pair:
                print(f"  Contact: {contact_pair}")
            else:
                print(f"  No contact pair available (run multi_protein_main.py first)")

        # Generate filename
        safe_name = protein_name.replace(" ", "_").lower()
        seq_suffix = "_corrupted" if is_corrupted else ""
        save_path = os.path.join(
            args.output_dir, f"{safe_name}_L{args.layer}H{args.head}{seq_suffix}.png"
        )

        plot_head_attention(
            model,
            alphabet,
            sequence,
            layer=args.layer,
            head=args.head,
            contact_pair=contact_pair,
            device=device,
            save_path=save_path,
            protein_name=protein_name,
            is_corrupted=is_corrupted,
        )

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
