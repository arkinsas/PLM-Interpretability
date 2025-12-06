from typing import Any, Callable, Dict, List

import torch

from hooks import PatchManager


def sweep_heads_once(
    patch_manager: PatchManager,
    num_layers: int,
    num_heads: int,
    clean_batch: Dict[str, torch.Tensor],
    corrupted_batch: Dict[str, torch.Tensor],
    forward_fn: Callable[[Dict[str, torch.Tensor]], Any],
    metric_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    """
    Sweep one head at a time and compute recovery scores.

    Args:
        patch_manager: PatchManager bound to the model.
        num_layers: total transformer layers.
        num_heads: heads per layer.
        clean_batch: inputs for the clean run.
        corrupted_batch: inputs for the corrupted run.
        forward_fn: callable to run the model, e.g., lambda b: model(**b).
        metric_fn: maps model output to a metric; may be scalar or per-example vector (higher is better).

    Returns:
        Dict with clean/corrupted metrics (mean + per-example), delta, and per-head results.
    """
    results: List[Dict[str, Any]] = []

    with torch.inference_mode():
        # Record clean activations and metric.
        with patch_manager.recording():
            clean_out = forward_fn(clean_batch)
        clean_metric_raw = metric_fn(clean_out)

        # Corrupted baseline metric.
        corrupted_out = forward_fn(corrupted_batch)
        corrupted_metric_raw = metric_fn(corrupted_out)

    def to_vec(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.flatten().detach().cpu()
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor([float(x)], dtype=torch.float32)

    clean_vec = to_vec(clean_metric_raw)
    corrupted_vec = to_vec(corrupted_metric_raw)

    clean_metric = float(clean_vec.mean().item())
    corrupted_metric = float(corrupted_vec.mean().item())

    delta_vec = clean_vec - corrupted_vec

    for li in range(num_layers):
        for hi in range(num_heads):
            patch_manager.set_target(heads={(li, hi)})
            with torch.inference_mode():
                with patch_manager.patching():
                    patched_out = forward_fn(corrupted_batch)
            patched_metric_raw = metric_fn(patched_out)
            patched_vec = to_vec(patched_metric_raw)
            patched_metric = float(patched_vec.mean().item())
            # Per-example recovery; avoid div by zero
            recovery_vec = torch.zeros_like(delta_vec)
            nonzero = delta_vec != 0
            recovery_vec[nonzero] = (
                patched_vec[nonzero] - corrupted_vec[nonzero]
            ) / delta_vec[nonzero]
            recovery_mean = float(recovery_vec.mean().item())
            results.append(
                {
                    "layer": li,
                    "head": hi,
                    "patched_metric_mean": patched_metric,
                    "patched_metric_per_example": patched_vec.tolist(),
                    "recovery": recovery_mean,
                    "recovery_per_example": recovery_vec.tolist(),
                }
            )

    return {
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "clean_metrics": clean_vec.tolist(),
        "corrupted_metrics": corrupted_vec.tolist(),
        "delta": (clean_metric - corrupted_metric),
        "delta_per_example": delta_vec.tolist(),
        "head_results": results,
    }


def patch_single_head(
    patch_manager: PatchManager,
    layer: int,
    head: int,
    clean_batch: Dict[str, torch.Tensor],
    corrupted_batch: Dict[str, torch.Tensor],
    forward_fn: Callable[[Dict[str, torch.Tensor]], Any],
    metric_fn: Callable[[Any], float],
) -> Dict[str, float]:
    """
    Patch a single (layer, head) pair and report recovery metrics.

    Returns:
        Dict with clean_metric, corrupted_metric, patched_metric, and recovery.
    """
    with torch.inference_mode():
        with patch_manager.recording():
            clean_out = forward_fn(clean_batch)
        clean_metric = float(metric_fn(clean_out))

        corrupted_out = forward_fn(corrupted_batch)
        corrupted_metric = float(metric_fn(corrupted_out))

    delta = clean_metric - corrupted_metric

    patch_manager.set_target(heads={(layer, head)})
    with torch.inference_mode():
        with patch_manager.patching():
            patched_out = forward_fn(corrupted_batch)
    patched_metric = float(metric_fn(patched_out))

    recovery = 0.0
    if delta != 0.0:
        recovery = (patched_metric - corrupted_metric) / delta

    return {
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "patched_metric": patched_metric,
        "recovery": recovery,
    }
