import numpy as np
import torch


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    array = bits.detach().cpu().numpy().astype(np.uint8, copy=False)
    n = array.shape[-1]
    padding = (-n) % 64
    padded = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, padding)])
    grouped = padded.reshape(*padded.shape[:-1], padded.shape[-1] // 64, 64)
    powers = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
    packed = np.sum(grouped.astype(np.uint64) * powers, axis=-1, dtype=np.uint64)
    return torch.from_numpy(packed.view(np.int64).copy())


def dense(spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (spikes.to(torch.int64) @ weights.to(torch.int64).T).to(torch.int32)


def lif_step(
    input_current: torch.Tensor,
    membrane: torch.Tensor,
    decay_q: int,
    threshold: int,
    v_reset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = membrane.to(torch.int64)
    retained = torch.div(
        (previous - v_reset) * decay_q,
        32768,
        rounding_mode="trunc",
    )
    charged = v_reset + retained + input_current.to(torch.int64)
    bounded = charged.clamp(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
    )
    fired = bounded >= threshold
    next_membrane = torch.where(
        fired,
        torch.full_like(bounded, v_reset),
        bounded,
    ).to(torch.int32)
    return pack_bits(fired.to(torch.uint8)), next_membrane


def mlp(
    spikes: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    decay_q: int,
    threshold: int,
    v_reset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    membrane1 = torch.zeros(
        spikes.shape[1], weight1.shape[0], dtype=torch.int32
    )
    membrane2 = torch.zeros(
        spikes.shape[1], weight2.shape[0], dtype=torch.int32
    )
    outputs = []
    for step in spikes:
        current1 = dense(step, weight1)
        hidden, membrane1 = lif_step(
            current1, membrane1, decay_q, threshold, v_reset
        )
        hidden_bits = unpack_bits(hidden, weight1.shape[0])
        current2 = dense(hidden_bits, weight2)
        output, membrane2 = lif_step(
            current2, membrane2, decay_q, threshold, v_reset
        )
        outputs.append(output)
    return torch.stack(outputs), membrane1, membrane2


def unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    array = packed.detach().cpu().numpy().view(np.uint64)
    bits = np.right_shift(array[..., :, None], np.arange(64, dtype=np.uint64)) & 1
    flat = bits.reshape(*array.shape[:-1], array.shape[-1] * 64)
    return torch.from_numpy(flat[..., :n].astype(np.uint8).copy())
