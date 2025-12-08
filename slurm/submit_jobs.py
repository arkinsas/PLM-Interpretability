import argparse
import subprocess
from pathlib import Path

from protein_config import Protein


def create_slurm_script(
    protein_name,
    start_trial,
    end_trial,
    output_dir,
    slurm_dir,
    time_limit="2:00:00",
    mem="8G",
    partition="ice-gpu",
):
    """Create a SLURM batch script."""

    protein_safe = protein_name.replace(" ", "_").replace("-", "_")
    num_trials = end_trial - start_trial

    gres_line = ""
    if "gpu" in partition.lower():
        gres_line = f"#SBATCH --gres=gpu:1"

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=robust_{protein_safe}_{start_trial}
#SBATCH --output={slurm_dir}/logs/{protein_safe}_%A_%a.out
#SBATCH --error={slurm_dir}/logs/{protein_safe}_%A_%a.err
#SBATCH --array=0-{num_trials - 1}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=1
#SBATCH --partition={partition}
{gres_line}

cd $SLURM_SUBMIT_DIR

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

TRIAL_NUM=$(({start_trial} + $SLURM_ARRAY_TASK_ID))

python slurm/run_trial.py \\
    --protein "{protein_name}" \\
    --trial $TRIAL_NUM \\
    --output-dir {output_dir}
"""

    script_path = slurm_dir / f"run_{protein_safe}_{start_trial}.sh"
    with open(script_path, "w") as f:
        f.write(slurm_script)
    script_path.chmod(0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(description="Submit robustness test jobs")
    parser.add_argument("--start-trial", type=int, default=0, help="Start trial number")
    parser.add_argument("--end-trial", type=int, required=True, help="End trial number (exclusive)")
    parser.add_argument("--protein", type=str, help="Single protein (default: all)")
    parser.add_argument("--output-dir", type=str, default="results/robustness_trials")
    parser.add_argument("--slurm-dir", type=str, default="slurm/scripts")
    parser.add_argument("--time-limit", type=str, default="2:00:00")
    parser.add_argument("--mem", type=str, default="8G")
    parser.add_argument("--partition", type=str, default="ice-gpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    num_trials = args.end_trial - args.start_trial

    if num_trials > 500:
        print(f"ERROR: Cannot submit {num_trials} trials (max 500 per batch)")
        print(f"Submit in batches:")
        print(f"  python slurm/submit_jobs.py --start-trial {args.start_trial} --end-trial {args.start_trial + 500}")
        print(f"  python slurm/submit_jobs.py --start-trial {args.start_trial + 500} --end-trial {args.end_trial}")
        return 1

    # Setup
    slurm_dir = Path(args.slurm_dir)
    slurm_dir.mkdir(exist_ok=True)
    (slurm_dir / "logs").mkdir(exist_ok=True)

    if args.protein:
        proteins = [args.protein]
    else:
        proteins = [name for name, _ in Protein.get_all_proteins()]

    total_jobs = len(proteins) * num_trials
    if total_jobs > 500:
        print(f"ERROR: {len(proteins)} proteins × {num_trials} trials = {total_jobs} jobs (max 500)")
        print(f"Submit one protein at a time or reduce trials")
        return 1

    print(f"Submitting: {len(proteins)} proteins, trials {args.start_trial}-{args.end_trial - 1} ({num_trials} each)")
    print(f"Total jobs: {total_jobs}\n")

    for protein_name in proteins:
        script_path = create_slurm_script(
            protein_name,
            args.start_trial,
            args.end_trial,
            args.output_dir,
            slurm_dir,
            args.time_limit,
            args.mem,
            args.partition,
        )

        if args.dry_run:
            print(f"{protein_name}: Created {script_path}")
        else:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"{protein_name}: Submitted job {job_id}")
            else:
                print(f"{protein_name}: ERROR - {result.stderr}")

    print(f"\nMonitor: squeue -u $USER")


if __name__ == "__main__":
    main()
