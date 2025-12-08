import json
import os
from pathlib import Path

import esm
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_head_attention(
    model,
    alphabet,
    sequence,
    layer,
    head,
    contact_pair,
    protein_name,
    device,
    save_path,
    is_corrupted=False,
):
    """
    Plots the attention heatmap for a specific Layer/Head with contact highlighted.

    Args:
        is_corrupted: Whether this is a corrupted sequence (for title annotation)
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        # Get attention weights
        results = model(batch_tokens, repr_layers=[], need_head_weights=True)
        attn = results["attentions"]

        # Handle different tensor shapes
        num_layers = len(model.layers)

        if attn.shape[0] == num_layers:
            # Layers are dim 0: (Layers, Batch, Heads, Seq, Seq)
            attn_matrix = attn[layer, 0, head].cpu().numpy()
        elif attn.shape[1] == num_layers:
            # Layers are dim 1: (Batch, Layers, Heads, Seq, Seq)
            attn_matrix = attn[0, layer, head].cpu().numpy()
        else:
            raise ValueError(
                f"Could not determine layer dimension. Shape: {attn.shape}, Layers: {num_layers}"
            )

    # Create plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attn_matrix, cmap="viridis", square=True, cbar_kws={"shrink": 0.8})

    seq_type = "CORRUPTED" if is_corrupted else "Clean"
    plt.title(
        f"{protein_name} [{seq_type}] - Layer {layer} Head {head}\nContact: {contact_pair}"
    )
    plt.xlabel("Key Position (Source)")
    plt.ylabel("Query Position (Destination)")

    # Highlight the contact points
    i, j = contact_pair
    # Draw red boxes at (i,j) and (j,i)
    rect1 = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2.5)
    rect2 = plt.Rectangle((i, j), 1, 1, fill=False, edgecolor="red", lw=2.5)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    """Generate visualizations using exact experimental setup from multi_protein_main.py."""
    # Load experimental metadata
    metadata_path = "results/experimental_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"ERROR: {metadata_path} not found!")
        print("Please run multi_protein_main.py first to generate experimental metadata.")
        return

    with open(metadata_path, "r") as f:
        experimental_metadata = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading ESM-2 model (esm2_t6_8M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    num_layers = len(model.layers)
    num_heads = model.layers[0].self_attn.num_heads

    print(f"Model loaded: {num_layers} layers, {num_heads} heads per layer")
    print(f"Loaded metadata for {len(experimental_metadata)} proteins")
    print(f"Total visualizations: {len(experimental_metadata) * num_layers * num_heads}")

    # Create output directory
    output_base = Path("visualizations_corrupted")
    output_base.mkdir(exist_ok=True)

    # Process each protein from experimental metadata
    total_count = 0
    for protein_name, metadata in experimental_metadata.items():
        print(f"\n{'='*60}")
        print(f"Processing: {protein_name}")
        print(f"Contact: {tuple(metadata['contact_pair'])}")
        print(f"Corruption: {metadata['corruption_method']}")
        print(f"Clean score: {metadata['clean_score']:.4f}")
        print(f"Corrupted score: {metadata['corrupted_score']:.4f}")
        print(f"{'='*60}")

        contact_pair = tuple(metadata["contact_pair"])
        corrupted_sequence = metadata["corrupted_sequence"]

        # Create protein-specific directory
        protein_dir = output_base / protein_name.replace(" ", "_").lower()
        protein_dir.mkdir(exist_ok=True)

        # Save metadata info for reference
        info_path = protein_dir / "experiment_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Protein: {protein_name}\n")
            f.write(f"Contact Pair: {contact_pair}\n")
            f.write(f"Corruption Method: {metadata['corruption_method']}\n")
            f.write(f"Clean Score: {metadata['clean_score']:.4f}\n")
            f.write(f"Corrupted Score: {metadata['corrupted_score']:.4f}\n")
            f.write(f"\nTop 10 Heads by Recovery:\n")
            if "top_heads" in metadata:
                for i, head in enumerate(metadata["top_heads"], 1):
                    f.write(
                        f"  {i}. Layer {head['layer']} Head {head['head']}: "
                        f"Recovery {head['recovery']:.4f}\n"
                    )

        # Generate visualization for each layer and head
        for layer in range(num_layers):
            for head in range(num_heads):
                save_path = protein_dir / f"L{layer:02d}H{head:02d}.png"

                plot_head_attention(
                    model=model,
                    alphabet=alphabet,
                    sequence=corrupted_sequence,  # Use CORRUPTED sequence
                    layer=layer,
                    head=head,
                    contact_pair=contact_pair,
                    protein_name=protein_name,
                    device=device,
                    save_path=save_path,
                    is_corrupted=True,  # Mark as corrupted in title
                )

                total_count += 1

                # Progress indicator
                if (head + 1) % 5 == 0:
                    print(f"  Layer {layer}: Completed {head + 1}/{num_heads} heads")

        print(f"✓ {protein_name} complete ({num_layers * num_heads} visualizations)")

    print(f"\n{'='*60}")
    print(f"ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Total visualizations generated: {total_count}")
    print(f"Output directory: {output_base}/")
    print(f"\nEach protein directory contains:")
    print(f"  - experiment_info.txt (metadata and top heads)")
    print(f"  - L00H00.png ... L{num_layers-1:02d}H{num_heads-1:02d}.png (attention maps)")
    print(
        f"\nThese visualizations use the CORRUPTED sequences from the causal tracing experiments."
    )


if __name__ == "__main__":
    main()
