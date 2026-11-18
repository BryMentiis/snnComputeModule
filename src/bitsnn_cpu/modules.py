from typing import Literal

import torch
from torch import nn

from .ops import DenseBackend, binary_dense, lif_step, pack_weights


class PackedBitLinear(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        packed_weight: torch.Tensor | None = None,
        backend: DenseBackend = "auto",
    ) -> None:
        super().__init__()
        if n_in <= 0 or n_out <= 0:
            raise ValueError("n_in and n_out must be positive")
        if backend not in ("auto", "scalar", "avx2"):
            raise ValueError(f"unknown backend: {backend}")
        self.n_in = n_in
        self.n_out = n_out
        self.backend = backend
        words = (n_in + 63) // 64
        if packed_weight is None:
            packed_weight = torch.zeros(n_out, words, dtype=torch.int64)
        if packed_weight.shape != (n_out, words):
            raise ValueError(
                f"packed_weight must have shape {(n_out, words)}, got {tuple(packed_weight.shape)}"
            )
        self.register_buffer("packed_weight", packed_weight.detach().clone().contiguous())

    @classmethod
    def from_binary(
        cls, weights: torch.Tensor, backend: DenseBackend = "auto"
    ) -> "PackedBitLinear":
        if weights.ndim != 2:
            raise ValueError("weights must have shape [Nout, Nin]")
        return cls(
            n_in=weights.shape[1],
            n_out=weights.shape[0],
            packed_weight=pack_weights(weights),
            backend=backend,
        )

    def forward(self, packed_spikes: torch.Tensor) -> torch.Tensor:
        return binary_dense(packed_spikes, self.packed_weight, self.n_in, self.backend)

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, n_out={self.n_out}, backend={self.backend}"


class BitLIFNode(nn.Module):
    def __init__(
        self,
        decay_q: int,
        threshold: int,
        v_reset: int = 0,
        step_mode: Literal["s", "m"] = "s",
    ) -> None:
        super().__init__()
        if not 0 <= decay_q <= 32768:
            raise ValueError("decay_q must be in [0, 32768]")
        self.decay_q = decay_q
        self.threshold = threshold
        self.v_reset = v_reset
        self.step_mode = step_mode
        self.v: torch.Tensor | None = None

    @classmethod
    def from_tau(
        cls,
        tau: float,
        threshold: int,
        v_reset: int = 0,
        step_mode: Literal["s", "m"] = "s",
    ) -> "BitLIFNode":
        if tau < 1.0:
            raise ValueError("tau must be at least 1")
        decay_q = round((1.0 - 1.0 / tau) * 32768)
        return cls(decay_q, threshold, v_reset, step_mode)

    def set_step_mode(self, step_mode: Literal["s", "m"]) -> None:
        if step_mode not in ("s", "m"):
            raise ValueError("step_mode must be 's' or 'm'")
        self.step_mode = step_mode

    def reset(self) -> None:
        self.v = None

    def _single_step(self, input_current: torch.Tensor) -> torch.Tensor:
        if input_current.ndim != 2:
            raise ValueError("single-step input must have shape [B, N]")
        if self.v is None:
            self.v = torch.zeros_like(input_current, dtype=torch.int32)
        if self.v.shape != input_current.shape:
            raise ValueError("input shape changed without calling reset()")
        spikes, self.v = lif_step(
            input_current,
            self.v,
            self.decay_q,
            self.threshold,
            self.v_reset,
        )
        return spikes

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._single_step(input_current)
        if input_current.ndim != 3:
            raise ValueError("multi-step input must have shape [T, B, N]")
        return torch.stack([self._single_step(step) for step in input_current], dim=0)

    def extra_repr(self) -> str:
        return (
            f"decay_q={self.decay_q}, threshold={self.threshold}, "
            f"v_reset={self.v_reset}, step_mode={self.step_mode}"
        )
