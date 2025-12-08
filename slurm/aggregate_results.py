import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from protein_config import Protein


def aggregate_protein_results(protein_name, trial_files, output_format="json"):
    """Aggregate all trials for a single protein."""
    if not trial_files:
        return None

    # Collect data across all trials
    head_counter = Counter()
    head_recoveries = defaultdict(list)
    layer_recoveries = defaultdict(list)
    contact_pairs = set()
    clean_scores = []
    corrupted_scores = []

    for trial_file in trial_files:
        with open(trial_file, "r") as f:
            trial_data = json.load(f)

        contact_pairs.add(tuple(trial_data["contact_pair"]))
        clean_scores.append(trial_data["clean_score"])
        corrupted_scores.append(trial_data["corrupted_score"])

        # Head statistics
        for rank, head_info in enumerate(trial_data["top_10_heads"], 1):
            head_key = f"L{head_info['layer']}H{head_info['head']}"
            head_counter[head_key] += 1
            head_recoveries[head_key].append(
                {"recovery": head_info["recovery"], "rank": rank}
            )

        # Layer statistics
        for layer_info in trial_data["layer_results"]:
            layer_key = f"L{layer_info['layer']}"
            layer_recoveries[layer_key].append(layer_info["recovery"])

    num_trials = len(trial_files)

    # Compute head statistics
    top_10_heads = head_counter.most_common(10)
    head_stats = {
        head: {
            "appearances": count,
            "appearance_rate": count / num_trials,
            "avg_recovery": sum(r["recovery"] for r in head_recoveries[head])
            / len(head_recoveries[head]),
            "std_recovery": _std(
                [r["recovery"] for r in head_recoveries[head]]
            ),
            "avg_rank": sum(r["rank"] for r in head_recoveries[head])
            / len(head_recoveries[head]),
        }
        for head, count in top_10_heads
    }

    # Compute layer statistics
    layer_stats = {}
    for layer_key in sorted(layer_recoveries.keys(), key=lambda x: int(x[1:])):
        recoveries = layer_recoveries[layer_key]
        layer_stats[layer_key] = {
            "num_observations": len(recoveries),
            "mean_recovery": sum(recoveries) / len(recoveries),
            "std_recovery": _std(recoveries),
            "min_recovery": min(recoveries),
            "max_recovery": max(recoveries),
        }

    summary = {
        "protein_name": protein_name,
        "num_trials": num_trials,
        "contact_pairs": sorted([list(cp) for cp in contact_pairs]),
        "avg_clean_score": sum(clean_scores) / len(clean_scores),
        "avg_corrupted_score": sum(corrupted_scores) / len(corrupted_scores),
        "head_statistics": head_stats,
        "layer_statistics": layer_stats,
    }

    return summary


def _std(values):
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5


def save_summary_json(summary, output_file):
    """Save summary as JSON."""
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)


def save_summary_csv(summary, output_file):
    """Save summary as CSV (separate files for heads and layers)."""
    base_path = output_file.with_suffix("")
    protein_name = summary["protein_name"]

    # Save head statistics
    heads_file = base_path.with_name(f"{base_path.name}_heads.csv")
    with open(heads_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "head",
                "appearances",
                "appearance_rate",
                "avg_recovery",
                "std_recovery",
                "avg_rank",
            ]
        )
        for head, stats in summary["head_statistics"].items():
            writer.writerow(
                [
                    head,
                    stats["appearances"],
                    stats["appearance_rate"],
                    stats["avg_recovery"],
                    stats["std_recovery"],
                    stats["avg_rank"],
                ]
            )

    # Save layer statistics
    layers_file = base_path.with_name(f"{base_path.name}_layers.csv")
    with open(layers_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "num_observations",
                "mean_recovery",
                "std_recovery",
                "min_recovery",
                "max_recovery",
            ]
        )
        for layer, stats in summary["layer_statistics"].items():
            writer.writerow(
                [
                    layer,
                    stats["num_observations"],
                    stats["mean_recovery"],
                    stats["std_recovery"],
                    stats["min_recovery"],
                    stats["max_recovery"],
                ]
            )

    # Save metadata
    meta_file = base_path.with_name(f"{base_path.name}_metadata.csv")
    with open(meta_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["protein_name", summary["protein_name"]])
        writer.writerow(["num_trials", summary["num_trials"]])
        writer.writerow(["avg_clean_score", summary["avg_clean_score"]])
        writer.writerow(["avg_corrupted_score", summary["avg_corrupted_score"]])
        writer.writerow(
            ["contact_pairs", str(summary["contact_pairs"])]
        )

    return [heads_file, layers_file, meta_file]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate robustness test results"
    )
    parser.add_argument(
        "--trials-dir",
        type=str,
        default="results/robustness_trials",
        help="Directory containing trial result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/robustness_summaries",
        help="Directory for aggregated results",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format (json or csv)",
    )
    parser.add_argument(
        "--protein",
        type=str,
        help="Aggregate specific protein only (default: all proteins)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Expected number of trials (for validation)",
    )
    args = parser.parse_args()

    trials_dir = Path(args.trials_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which proteins to aggregate
    if args.protein:
        proteins_to_aggregate = [args.protein]
    else:
        proteins_to_aggregate = [name for name, _ in Protein.get_all_proteins()]

    print(f"Aggregating robustness test results")
    print(f"  Trial directory: {trials_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output format: {args.output_format}")
    print()

    summaries = {}

    for protein_name in proteins_to_aggregate:
        print(f"Processing {protein_name}...", end=" ")

        # Find all trial files for this protein
        trial_files = sorted(trials_dir.glob(f"{protein_name}_trial_*.json"))

        if not trial_files:
            print(f"No trial files found")
            continue

        print(f"found {len(trial_files)} trials")

        if args.trials and len(trial_files) != args.trials:
            print(
                f"  WARNING: Expected {args.trials} trials, found {len(trial_files)}"
            )

        # Aggregate results
        summary = aggregate_protein_results(
            protein_name, trial_files, args.output_format
        )

        if summary:
            summaries[protein_name] = summary

            # Save per-protein file
            num_trials = len(trial_files)
            output_file = (
                output_dir / f"robustness_{num_trials}_{protein_name}.{args.output_format}"
            )

            if args.output_format == "json":
                save_summary_json(summary, output_file)
                print(f"  Saved: {output_file}")
            else:
                csv_files = save_summary_csv(summary, output_file)
                print(f"  Saved: {', '.join(str(f) for f in csv_files)}")

            # Print key statistics
            top_head = list(summary["head_statistics"].items())[0]
            top_layer = max(
                summary["layer_statistics"].items(),
                key=lambda x: x[1]["mean_recovery"],
            )
            print(
                f"  Top head: {top_head[0]} ({top_head[1]['appearance_rate']:.1%} appearance)"
            )
            print(
                f"  Top layer: {top_layer[0]} (recovery: {top_layer[1]['mean_recovery']:.3f})"
            )

    print()
    print(f"Aggregation complete: {len(summaries)} proteins processed")


if __name__ == "__main__":
    main()
