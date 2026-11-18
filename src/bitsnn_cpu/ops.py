from typing import Literal

import torch


DenseBackend = Literal["auto", "scalar", "avx2"]


def pack_spikes(spikes: torch.Tensor) -> torch.Tensor:
    return torch.ops.bitsnn_cpu.pack_spikes(spikes.contiguous())


def pack_weights(weights: torch.Tensor) -> torch.Tensor:
    return torch.ops.bitsnn_cpu.pack_weights(weights.contiguous())


def unpack_spikes(packed: torch.Tensor, n: int) -> torch.Tensor:
    return torch.ops.bitsnn_cpu.unpack_spikes(packed.contiguous(), n)


def binary_dense(
    spikes: torch.Tensor,
    weights: torch.Tensor,
    n_in: int,
    backend: DenseBackend = "auto",
) -> torch.Tensor:
    spikes = spikes.contiguous()
    weights = weights.contiguous()
    if backend == "auto":
        return torch.ops.bitsnn_cpu.binary_dense(spikes, weights, n_in)
    if backend == "scalar":
        return torch.ops.bitsnn_cpu.binary_dense_scalar(spikes, weights, n_in)
    if backend == "avx2":
        return torch.ops.bitsnn_cpu.binary_dense_avx2(spikes, weights, n_in)
    raise ValueError(f"unknown backend: {backend}")


def binary_dense_unpacked(spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.ops.bitsnn_cpu.binary_dense_unpacked(
        spikes.contiguous(), weights.contiguous()
    )


def lif_step(
    input_current: torch.Tensor,
    membrane: torch.Tensor,
    decay_q: int,
    threshold: int,
    v_reset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = torch.ops.bitsnn_cpu.lif_step(
        input_current.contiguous(),
        membrane.contiguous(),
        decay_q,
        threshold,
        v_reset,
    )
    return result[0], result[1]


def avx2_available() -> bool:
    return bool(torch.ops.bitsnn_cpu.avx2_available())
