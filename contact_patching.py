import torch

from activation_patching import sweep_heads_once
from hooks import PatchManager


def run_contact_sweep(
    model,
    alphabet,
    device: torch.device,
    pairs: list[tuple[str, str, tuple[int, int]]],
    batch_size: int = 4,
):
    batch_converter = alphabet.get_batch_converter()

    pm = PatchManager(model, device)
    num_layers = len(model.layers)
    num_heads = model.layers[0].self_attn.num_heads

    pair_results = [
        {
            "index": idx,
            "clean": c,
            "corrupted": d,
            "contact_idx": c_idx,
            "head_results": [],
        }
        for idx, (c, d, c_idx) in enumerate(pairs)
    ]

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        clean_seqs = [p[0] for p in chunk]
        corrupted_seqs = [p[1] for p in chunk]

        contact_indices = [p[2] for p in chunk]

        _, _, clean_tokens = batch_converter(
            [(f"c{i}", s) for i, s in enumerate(clean_seqs)]
        )

        _, _, corrupted_tokens = batch_converter(
            [(f"d{i}", s) for i, s in enumerate(corrupted_seqs)]
        )

        clean_tokens = clean_tokens.to(device)
        corrupted_tokens = corrupted_tokens.to(device)

        def forward_fn(batch):
            # Output shape: (Batch, Seq_Len, Seq_Len)
            return model(batch["tokens"], return_contacts=True)["contacts"]

        def metric_fn(contact_out):
            vals = []
            for b, (residue_i, residue_j) in enumerate(contact_indices):
                # Extract the probability that residue_i and residue_j are in contact
                prob = contact_out[b, residue_i, residue_j].item()
                vals.append(prob)
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

        # Map results back to the global result list
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

    return {"pairs": pair_results}
