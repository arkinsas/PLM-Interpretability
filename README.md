# Mechanistic Interpretability for ESM-2 Protein Language Models

Activation patching to identify which attention heads and MLP layers in ESM-2 are causally responsible for predicting long-range protein contacts.

## Quick Start

```bash
# Setup
uv sync

# Single protein analysis
python main.py

# Multi-protein analysis (find universal heads)
python multi_protein_main.py

# Attention visualizations
python visualize.py
```

## Robustness Testing on PACE

Run hundreds of corruption trials in parallel.

```bash
# Submit 500 trials for one protein
python slurm/submit_jobs.py --protein Ubiquitin --start-trial 0 --end-trial 500

# Monitor and aggregate
squeue -u $USER
python slurm/aggregate_results.py
```

**Output:**
- Individual trials: `results/robustness_trials/Ubiquitin_trial_00000.json`
- Aggregated summaries: `results/robustness_summaries/robustness_500_Ubiquitin.json`

## Project Structure

**Core Analysis:**
- `main.py` - Single protein activation patching pipeline
- `multi_protein_main.py` - Cross-protein analysis to find universal heads
- `activation_patching.py` - Head-level patching (sweep all attention heads)
- `contact_patching.py` - Contact-based patching wrapper for protein pairs
- `hooks.py` - PatchManager for recording/patching activations via PyTorch hooks

**Utilities:**
- `protein_config.py` - Protein sequences and configurations (8 test proteins)
- `utils.py` - Corruption functions (shuffled/swapped sequence generation)
- `load_esm2.py` - ESM-2 model loader utility

**Visualization:**
- `visualize.py` - Generate attention heatmaps for specific heads
- `visualize_all_heads.py` - Generate visualizations for all heads across all proteins

**SLURM Robustness Testing:**
- `slurm/submit_jobs.py` - Submit parallel SLURM jobs (respects PACE-ICE 500 job limit)
- `slurm/run_trial.py` - Run single robustness trial (called by SLURM)
- `slurm/aggregate_results.py` - Aggregate trial results into summary statistics

## Models Supported

ESM-2 models (8M to 15B parameters):
- `esm2_t6_8M_UR50D`
- `esm2_t12_35M_UR50D`
- `esm2_t30_150M_UR50D`
- `esm2_t33_650M_UR50D`
- `esm2_t36_3B_UR50D`
- `esm2_t48_15B_UR50D`

## Dependencies

- PyTorch 2.9.1+
- fair-esm 2.0.0+
- matplotlib, seaborn
- numpy

## Test Proteins

8 well-characterized proteins: Ubiquitin (76aa), Protein G (56aa), SH3 Domain (57aa), WW Domain (34aa), Villin Headpiece (35aa), Trp-cage (20aa), Homeodomain (60aa), Zinc Finger (28aa)
