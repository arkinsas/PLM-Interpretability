"""Utilities for recording and patching ESM-2 activations.

PatchManager wraps PyTorch forward hooks to cache clean activations and
optionally swap them into a corrupted run, enabling activation patching
experiments without modifying the underlying model code.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import nn


@dataclass
class ActivationCache:
    """Holds cached activations from a clean forward pass."""

    attn: Dict[Tuple[int, int], torch.Tensor]  # (layer, head) -> tensor of shape (B, T, d_head)
    mlp: Dict[int, torch.Tensor]  # layer -> tensor of shape (B, T, D)


class PatchManager:
    """Manage forward hooks for recording and patching activations."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cache_device: Optional[torch.device] = None,
        cache_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model: ESM model (expects model.layers with self_attn and mlp).
            device: device where the model runs.
            cache_device: where to store cached activations ("cpu" by default; set to
                the model device to keep cache on GPU).
            cache_dtype: optional dtype for cached tensors (e.g., torch.float16 to
                reduce memory).
        """
        self.model = model
        self.device = device
        if cache_device is not None:
            self.cache_device = cache_device
        else:
            self.cache_device = torch.device("cpu")
        self.cache_dtype = cache_dtype
        self.mode: str = "off"  # "off" | "record" | "patch"
        self.cache: Optional[ActivationCache] = None
        self.target_layers: Set[int] = set()
        self.target_heads: Set[Tuple[int, int]] = set()
        self.target_mlp_layers: Set[int] = set()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def set_target(
        self,
        layers: Optional[Iterable[int]] = None,
        heads: Optional[Iterable[Tuple[int, int]]] = None,
        mlp_layers: Optional[Iterable[int]] = None,
    ) -> None:
        """Define which components should be patched."""
        self.target_layers = set(layers or [])
        self.target_heads = set(heads or [])
        self.target_mlp_layers = set(mlp_layers or [])

    def _clear_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    @contextmanager
    def recording(self):
        """Enable recording mode for a clean forward pass."""
        self.mode = "record"
        self.cache = ActivationCache(attn={}, mlp={})
        self._register_hooks()
        try:
            yield
        finally:
            self._clear_hooks()
            self.mode = "off"

    @contextmanager
    def patching(self):
        """Enable patching mode for a corrupted forward pass."""
        if self.cache is None:
            raise RuntimeError("Recording cache is empty. Run recording() first.")
        self.mode = "patch"
        self._register_hooks()
        try:
            yield
        finally:
            self._clear_hooks()
            self.mode = "off"

    def _register_hooks(self) -> None:
        """Attach forward hooks to attention and MLP submodules."""
        self._clear_hooks()
        for layer_idx, block in enumerate(self.model.layers):
            # Attention hook
            def make_attn_hook(li: int):
                def hook(mod: nn.Module, _inp, out):
                    is_tuple = isinstance(out, tuple)
                    attn_out = out[0] if is_tuple else out  # shape (B, T, D)
                    heads = mod.num_heads
                    d_head = attn_out.shape[-1] // heads
                    reshaped = attn_out.view(attn_out.shape[0], attn_out.shape[1], heads, d_head)

                    if self.mode == "record":
                        for h in range(heads):
                            cached = reshaped[:, :, h, :].detach()
                            if self.cache_dtype:
                                cached = cached.to(self.cache_dtype)
                            self.cache.attn[(li, h)] = cached.to(self.cache_device)
                    elif self.mode == "patch":
                        for h in range(heads):
                            key = (li, h)
                            if key in self.target_heads or li in self.target_layers:
                                cached = self.cache.attn.get(key)
                                if cached is not None:
                                    reshaped[:, :, h, :] = cached.to(reshaped.device)

                    patched = reshaped.view_as(attn_out)
                    if is_tuple:
                        # Preserve any auxiliary outputs (e.g., attention weights)
                        rest = out[1:]
                        return (patched, *rest)
                    return patched

                return hook

            self._hooks.append(block.self_attn.register_forward_hook(make_attn_hook(layer_idx)))

            # MLP hook
            mlp_module = getattr(block, "mlp", None)
            if mlp_module is None:
                mlp_module = getattr(block, "fc2", None)  # fair-esm uses fc2 as the second FFN linear
            if mlp_module is not None:
                def make_mlp_hook(li: int):
                    def hook(_mod: nn.Module, _inp, out):
                        if self.mode == "record":
                            cached = out.detach()
                            if self.cache_dtype:
                                cached = cached.to(self.cache_dtype)
                            self.cache.mlp[li] = cached.to(self.cache_device)
                        elif self.mode == "patch":
                            if li in self.target_mlp_layers or li in self.target_layers:
                                cached = self.cache.mlp.get(li)
                                if cached is not None:
                                    return cached.to(out.device)
                        return out

                    return hook

                self._hooks.append(mlp_module.register_forward_hook(make_mlp_hook(layer_idx)))
