import esm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from hooks import PatchManager
from main import run_single_protein_analysis
from protein_config import Protein
from utils import create_corrupted_shuffled, create_corrupted_swapped


def get_best_contact(model, alphabet, sequence, device):
    """Finds the strongest long-range contact."""
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, return_contacts=True)
        contacts = results["contacts"][0]

    L = contacts.shape[0]
    candidates = []

    # Scan for contacts
    min_sep = 10
    for i in range(1, L - 1):
        for j in range(i + min_sep, L - 1):
            score = contacts[i, j].item()
            if score > 0.6:
                candidates.append(((i, j), score, j - i))

    if not candidates:
        return None, None

    # Sort by Distance first, then Score.
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

    return candidates[0][0], candidates[0][1]


def run_experiment(model, alphabet, device, name, seq):
    print(f"Processing {name} (Len: {len(seq)})...")

    result = get_best_contact(model, alphabet, seq, device)

    if result[0] is None:
        print("  -> Skipping (No strong long-range contacts > 10 residues)")
        return None

    (t_i, t_j), clean_score = result
    print(f"  -> Target: ({t_i}, {t_j}) Dist: {t_j - t_i} Score: {clean_score:.2f}")

    #corr_seq = create_corrupted_shuffled(seq, t_i, t_j)
    corr_seq = create_corrupted_swapped(seq, t_i, t_j, Protein.get_all_proteins())

    batch_converter = alphabet.get_batch_converter()
    _, _, clean_toks = batch_converter([("c", seq)])
    _, _, corr_toks = batch_converter([("d", corr_seq)])
    clean_dict = {"tokens": clean_toks.to(device)}
    corr_dict = {"tokens": corr_toks.to(device)}

    pm = PatchManager(model, device)

    with torch.no_grad():
        with pm.recording():
            model(clean_dict["tokens"], return_contacts=True)

        res_corr = model(corr_dict["tokens"], return_contacts=True)
        corr_score = res_corr["contacts"][0, t_i, t_j].item()

    denom = clean_score - corr_score
    if denom < 0.1:
        print(f"  -> Skipping (Signal differential too weak: {denom:.2f})")
        return None

    # Sweep MLP Layers
    layer_recoveries = []
    num_layers = len(model.layers)

    for layer_idx in range(num_layers):
        pm.set_target(mlp_layers={layer_idx})
        with torch.no_grad():
            with pm.patching():
                res = model(corr_dict["tokens"], return_contacts=True)
                patched_score = res["contacts"][0, t_i, t_j].item()

        recovery = (patched_score - corr_score) / denom
        layer_recoveries.append(recovery)

    print(f"  -> L4 Recovery: {layer_recoveries[4]:.2f}")
    return layer_recoveries


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading ESM-2...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    mlp_results = []
    leaderboard = {}

    print("STARTING MULTI-PROTEIN SWEEP")

    for name, seq in Protein.get_all_proteins():
        res = run_single_protein_analysis(model, alphabet, device, name, seq)

        # Find the universal head
        if res:
            top_head = sorted(
                res["head_results"], key=lambda x: x["recovery"], reverse=True
            )[0]
            key = f"L{top_head['layer']}H{top_head['head']}"

            if key not in leaderboard:
                leaderboard[key] = []
            leaderboard[key].append(name)

            print(f"WINNER for {name}:{key} (Recovery: {top_head['recovery']:.2f})")

        # Complete the MLP sweep for the boxplot
        mlp_res = run_experiment(model, alphabet, device, name, seq)
        if mlp_res:
            mlp_results.append(mlp_res)
        else:
            print(f"{name} skipped for MLP plot")

    print("FINAL HEAD RESULTS -> UNIVERSAL HEADS")
    sorted_leaders = sorted(leaderboard.items(), key=lambda x: len(x[1]), reverse=True)

    # Print the number of causalities per head
    for head, proteins in sorted_leaders:
        print(f"HEAD {head} | Wins: {len(proteins)} | Proteins: {proteins}")

    # Convert to Numpy
    data = np.array(mlp_results)
    num_layers = data.shape[1]

    # Print Stats
    print("\n" + "=" * 30)
    print(f"AGGREGATE RESULTS (N={len(mlp_results)})")
    print("=" * 30)
    print("Layer | Mean Rec | Std Dev")
    print("-" * 25)
    for layer in range(num_layers):
        print(
            f"MLP {layer} | {data[:, layer].mean():7.2f}  | {data[:, layer].std():5.2f}"
        )

    # Plotting
    plt.figure(figsize=(10, 6))

    # Create Boxplot
    sns.boxplot(data=data, color="lightblue")
    sns.swarmplot(data=data, color=".25")

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axhline(1, color="green", linestyle="--", linewidth=1, alpha=0.5)

    plt.title(f"Mechanism Consistency Across {len(mlp_results)} Proteins (ESM-2 8M)")
    plt.ylabel("Recovery Score (1.0 = Full Restoration)")
    plt.xlabel("MLP Layer Index")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
