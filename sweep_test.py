import random

import torch
from esm import pretrained

from sweep_runner import run_head_sweep


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = pretrained.esm2_t12_35M_UR50D()
    model = model.eval().to(device)

    # Build a small set of clean/corrupted pairs
    clean_seq = "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWGEDTLMEYLENPKKYIPGTKMIFAGIKKK"
    pairs = []
    for mask_pos in [15, 20]:
        span_start, span_end = 30, 40
        span = list(clean_seq[span_start:span_end])
        random.shuffle(span)
        corrupted_seq = clean_seq[:span_start] + "".join(span) + clean_seq[span_end:]
        pairs.append((clean_seq, corrupted_seq, mask_pos))

    res = run_head_sweep(model, alphabet, device, pairs, batch_size=2)

    # Print summary
    for pr in res["pairs"]:
        print(f"Pair {pr['index']} mask_pos={pr['mask_pos']}")
        print("  clean_metric:", pr["clean_metric"])
        print("  corrupted_metric:", pr["corrupted_metric"])
        print(
            "  top 3 heads:",
            sorted(pr["head_results"], key=lambda x: x["recovery"], reverse=True)[:3],
        )
    print("Top averaged heads:", res["averaged_head_recovery"][:5])


if __name__ == "__main__":
    main()
