# rl-plm


## Using the hook utilities

- `hooks.py` provides `PatchManager` to record activations from a clean run and patch them into a corrupted run.
- `activation_patching.py` has helpers: `patch_single_head` (one head) and `sweep_heads_once` (loop over all heads).

Basic flow:
```python
from hooks import PatchManager
from activation_patching import patch_single_head

pm = PatchManager(model, device)

# Run clean (records) and corrupted, then patch a single head
res = patch_single_head(
    patch_manager=pm,
    layer=0,
    head=0,
    clean_batch={"tokens": clean_tokens},
    corrupted_batch={"tokens": corrupted_tokens},
    forward_fn=lambda b: model(**b)["logits"],
    metric_fn=lambda logits: float(logits.mean()),  # replace with your metric
)
print(res)
```

To sweep all heads:
```python
from activation_patching import sweep_heads_once
pm = PatchManager(model, device)
res = sweep_heads_once(
    patch_manager=pm,
    num_layers=len(model.layers),
    num_heads=model.layers[0].self_attn.num_heads,
    clean_batch={"tokens": clean_tokens},
    corrupted_batch={"tokens": corrupted_tokens},
    forward_fn=lambda b: model(**b)["logits"],
    metric_fn=metric_fn,  # your scalar metric
)
print(res["head_results"])  # list of dicts with layer/head/recovery
```

## Running current activation patching

Made run_head_sweep(model, alphabet, device: torch.device, pairs: List[Tuple[str, str, int]], batch_size: int = 4). This runs the sweep_heads_once form activation_patching.py on the list of pairs passed in. Pairs should have format of (cleansequence, corruptedsequence, maskpos). It returns a dict with per-pair sweep results (pairs: clean/corrupted metrics and head recoveries for each input) and averaged_head_recovery (head recoveries averaged across all pairs). Once a system for making the list of pairs is created (list of valid protein sequences and functions to corrupt them), run_head_sweep can be used to get results. 

## current step by step

1. install fair-esm if not done already
 ```pip install fair-esm```
2. pre-load a model (optional, but it takes a long time to load the larger versions of esm2 so id suggest running it because itll store the model in a cache and it will be ready to use once pre-loaded)
```python load_esm2.py --model (choose between esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D)```
3. rn the only thing ready to run is sweep_test, which is an example of how to run the attention patching.
```python sweep_test.py```

## TO-DOs
-Make a list of valid protein strings to use as the clean sequences
-Create system for corrupting strings. Swaps, deletions etc
-use the above two to create a list of tuples of the following form (cleansequence, corruptedsequence, maskposition). The mask position will be the protein/letter that is hidden from esm2 and which it will try to predict. randomizing mask position makes the most senese i think
-store/analyze results

## uneccesary  to-dos
-we can also do other ways of selecting the head/heads that are to be patched. Rn it does one at a time. After the recovery scores for each head are computed individually could try activating the highest ones together to see if that helps more than individually. Or the lowest to see if together they are important even though their scores are low

- Only the attention outputs and the output of the NNs are patched. Could get more finegrained and patch attention weights or other parts

## miscellaneious
the sweep_test serves as a good example of how run the sweep runner. Also if making a new way of activation patching, sweep runner serves as a good example of a wrapper.