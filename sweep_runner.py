from typing import Dict, List, Tuple

import torch

from activation_patching import sweep_heads_once
from hooks import PatchManager


def run_head_sweep(
    model,
    alphabet,
    device: torch.device,
    pairs: List[Tuple[str, str, int]],
    batch_size: int = 4,
) -> Dict:
    """
    Run head sweeps over (clean, corrupted, mask_pos) tuples, batching for efficiency,
    while keeping per-pair results.

    Args:
        model: ESM-2 model (already on device).
        alphabet: ESM alphabet (for batch_converter, mask_idx).
        device: torch device the model is on.
        pairs: list of (clean_seq, corrupted_seq, mask_pos).
        batch_size: how many pairs per batch.

    Returns:
        Dict with per-pair sweep results and averaged head recoveries across pairs.
    """
    if len(pairs) == 0:
        raise ValueError("No pairs provided.")

    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    pm = PatchManager(model, device)
    num_layers = len(model.layers)
    num_heads = model.layers[0].self_attn.num_heads

    pair_results = [
        {"index": idx, "clean": c, "corrupted": d, "mask_pos": m, "head_results": []}
        for idx, (c, d, m) in enumerate(pairs)
    ]

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        clean_names = [f"clean_{i}" for i in range(len(chunk))]
        corr_names = [f"corrupted_{i}" for i in range(len(chunk))]
        clean_seqs = [p[0] for p in chunk]
        corrupted_seqs = [p[1] for p in chunk]
        mask_positions = [p[2] for p in chunk]

        _, _, clean_tokens = batch_converter(list(zip(clean_names, clean_seqs)))
        _, _, corrupted_tokens = batch_converter(list(zip(corr_names, corrupted_seqs)))

        targets = []
        for i, pos in enumerate(mask_positions):
            targets.append(clean_tokens[i, pos].item())
            clean_tokens[i, pos] = mask_idx
            corrupted_tokens[i, pos] = mask_idx

        clean_tokens = clean_tokens.to(device)
        corrupted_tokens = corrupted_tokens.to(device)

        def forward_fn(batch):
            return model(batch["tokens"], repr_layers=[], return_contacts=False)[
                "logits"
            ]

        def metric_fn(logits):
            vals = []
            for b, (pos, tgt) in enumerate(zip(mask_positions, targets)):
                vals.append(logits[b, pos, tgt].item())
            return vals

        res = sweep_heads_once(
            patch_manager=pm,
            num_layers=num_layers,
            num_heads=num_heads,
            clean_batch={"tokens": clean_tokens},
            corrupted_batch={"tokens": corrupted_tokens},
            forward_fn=forward_fn,
            metric_fn=metric_fn,
        )

        # Assign per-example results back to global pair indices
        for local_b, global_idx in enumerate(
            range(start, min(start + batch_size, len(pairs)))
        ):
            pair_results[global_idx]["clean_metric"] = res["clean_metrics"][local_b]
            pair_results[global_idx]["corrupted_metric"] = res["corrupted_metrics"][
                local_b
            ]
            for hr in res["head_results"]:
                pair_results[global_idx]["head_results"].append(
                    {
                        "layer": hr["layer"],
                        "head": hr["head"],
                        "recovery": hr["recovery_per_example"][local_b],
                        "patched_metric": hr["patched_metric_per_example"][local_b],
                    }
                )

    agg = {}
    for res in pair_results:
        for hr in res["head_results"]:
            key = (hr["layer"], hr["head"])
            agg.setdefault(key, []).append(hr["recovery"])

    averaged = [
        {"layer": k[0], "head": k[1], "recovery": sum(v) / len(v)}
        for k, v in agg.items()
    ]
    averaged.sort(key=lambda x: x["recovery"], reverse=True)

    return {"pairs": pair_results, "averaged_head_recovery": averaged}
