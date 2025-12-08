import argparse
import json
import os
from pathlib import Path

import esm
import torch

from contact_patching import run_contact_sweep
from hooks import PatchManager
from multi_protein_main import CORRUPTION_METHOD, get_best_contact
from protein_config import Protein
from utils import create_corrupted_shuffled, create_corrupted_swapped


def run_single_trial(
    model, alphabet, device, protein_name, sequence, trial_num, output_dir
):
    """Run one corruption trial and save results to disk."""
    # Get contact pair (same across trials for a given protein)
    result = get_best_contact(model, alphabet, sequence, device)
    if result[0] is None:
        return None

    (t_i, t_j), clean_score = result

    # Apply corruption (RANDOM - differs across trials)
    if CORRUPTION_METHOD == "shuffled":
        corrupted_seq = create_corrupted_shuffled(sequence, t_i, t_j)
    elif CORRUPTION_METHOD == "swapped":
        corrupted_seq = create_corrupted_swapped(
            sequence, t_i, t_j, Protein.get_all_proteins()
        )
    else:
        raise ValueError(f"Unknown corruption method: {CORRUPTION_METHOD}")

    # Prepare batches
    batch_converter = alphabet.get_batch_converter()
    _, _, clean_toks = batch_converter([("c", sequence)])
    _, _, corr_toks = batch_converter([("d", corrupted_seq)])
    clean_dict = {"tokens": clean_toks.to(device)}
    corr_dict = {"tokens": corr_toks.to(device)}

    pm = PatchManager(model, device)

    # Get clean and corrupted baseline scores
    with torch.no_grad():
        with pm.recording():
            res_clean = model(clean_dict["tokens"], return_contacts=True)
            clean_score_actual = res_clean["contacts"][0, t_i, t_j].item()

        res_corr = model(corr_dict["tokens"], return_contacts=True)
        corr_score = res_corr["contacts"][0, t_i, t_j].item()

    denom = clean_score_actual - corr_score

    # Head-level analysis
    pairs = [(sequence, corrupted_seq, (t_i, t_j))]
    head_res = run_contact_sweep(model, alphabet, device, pairs)
    pair_res = head_res["pairs"][0]

    # Get top 10 heads by recovery
    top_heads = sorted(
        pair_res["head_results"], key=lambda x: x["recovery"], reverse=True
    )[:10]

    # Layer-level analysis
    num_layers = len(model.layers)
    layer_results = []

    for layer_idx in range(num_layers):
        pm.set_target(mlp_layers={layer_idx})
        with torch.no_grad():
            with pm.patching():
                res = model(corr_dict["tokens"], return_contacts=True)
                patched_score = res["contacts"][0, t_i, t_j].item()

        recovery = (patched_score - corr_score) / denom if denom != 0 else 0.0
        layer_results.append(
            {
                "layer": layer_idx,
                "recovery": float(recovery),
                "patched_score": float(patched_score),
            }
        )

    # Save trial result
    trial_data = {
        "protein_name": protein_name,
        "trial_num": trial_num,
        "contact_pair": (t_i, t_j),
        "clean_score": float(clean_score_actual),
        "corrupted_score": float(corr_score),
        "corruption_method": CORRUPTION_METHOD,
        "top_10_heads": [
            {
                "layer": h["layer"],
                "head": h["head"],
                "recovery": h["recovery"],
                "patched_metric": h["patched_metric"],
            }
            for h in top_heads
        ],
        "layer_results": layer_results,
    }

    # Save to file
    output_file = output_dir / f"{protein_name}_trial_{trial_num:05d}.json"
    with open(output_file, "w") as f:
        json.dump(trial_data, f, indent=2)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run single robustness trial for SLURM parallelization"
    )
    parser.add_argument(
        "--protein",
        type=str,
        required=True,
        help="Protein name to test",
    )
    parser.add_argument(
        "--trial",
        type=int,
        required=True,
        help="Trial number (0-indexed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/robustness_trials",
        help="Directory to save trial results",
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)

    # Get protein sequence
    protein = next(p for p in Protein if p.protein_name == args.protein)
    sequence = protein.sequence

    # Run trial
    result_file = run_single_trial(
        model, alphabet, device, args.protein, sequence, args.trial, output_dir
    )

    if result_file:
        print(f"Trial {args.trial} completed: {result_file}")
    else:
        print(f"Trial {args.trial} skipped (no valid contact)")


if __name__ == "__main__":
    main()
