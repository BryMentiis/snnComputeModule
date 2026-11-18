import platform

import torch

from bitsnn_cpu import (
    avx2_available,
    binary_dense,
    binary_dense_unpacked,
    lif_step,
    pack_spikes,
    pack_weights,
    unpack_spikes,
)
from benchmarks.common import median_seconds, milliseconds


def torch_lif(
    current: torch.Tensor,
    membrane: torch.Tensor,
    decay_q: int = 16384,
    threshold: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    retained = torch.div(membrane * decay_q, 32768, rounding_mode="trunc")
    charged = retained + current
    bounded = charged.clamp(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
    )
    fired = bounded >= threshold
    next_membrane = torch.where(fired, torch.zeros_like(bounded), bounded).to(torch.int32)
    return fired.to(torch.uint8), next_membrane


def main() -> None:
    torch.manual_seed(11)
    torch.set_num_threads(1)
    batch = 4
    print(f"CPU: {platform.processor()}")
    print(f"PyTorch: {torch.__version__}; threads: {torch.get_num_threads()}; AVX2: {avx2_available()}")
    print("Each timing is a median; dense shape is [4, N] x [N, N].")

    for n in (64, 256, 1024, 4096):
        weights = (torch.rand(n, n) < 0.5).to(torch.uint8)
        packed_weights = pack_weights(weights)
        float_weights = weights.float()
        print(
            f"\nN={n}; spike storage uint8={batch * n} B, "
            f"packed={batch * ((n + 63) // 64) * 8} B; "
            f"weights uint8={weights.numel()} B, packed={packed_weights.numel() * 8} B"
        )
        for density in (0.01, 0.10, 0.50, 1.00):
            spikes = (torch.rand(batch, n) < density).to(torch.uint8)
            float_spikes = spikes.float()
            packed_spikes = pack_spikes(spikes)
            membrane = torch.zeros(batch, n, dtype=torch.int32)

            expected = binary_dense_unpacked(spikes, weights)
            scalar_output = binary_dense(packed_spikes, packed_weights, n, "scalar")
            auto_output = binary_dense(packed_spikes, packed_weights, n)
            torch_output = float_spikes @ float_weights.T
            if not torch.equal(scalar_output, expected) or not torch.equal(auto_output, expected):
                raise RuntimeError("packed dense output mismatch")
            if not torch.equal(torch_output, expected.float()):
                raise RuntimeError("PyTorch dense output mismatch")

            def unpacked_dense() -> torch.Tensor:
                return binary_dense_unpacked(spikes, weights)

            def scalar_dense() -> torch.Tensor:
                return binary_dense(packed_spikes, packed_weights, n, "scalar")

            def auto_dense() -> torch.Tensor:
                return binary_dense(packed_spikes, packed_weights, n)

            def pytorch_dense() -> torch.Tensor:
                return float_spikes @ float_weights.T

            def packed_full() -> tuple[torch.Tensor, torch.Tensor]:
                packed_input = pack_spikes(spikes)
                current = binary_dense(packed_input, packed_weights, n)
                packed_output, next_membrane = lif_step(current, membrane, 16384, 32)
                return unpack_spikes(packed_output, n), next_membrane

            def pytorch_full() -> tuple[torch.Tensor, torch.Tensor]:
                current = (float_spikes @ float_weights.T).to(torch.int64)
                return torch_lif(current, membrane.to(torch.int64))

            full_spikes, full_membrane = packed_full()
            reference_spikes, reference_membrane = pytorch_full()
            if not torch.equal(full_spikes, reference_spikes) or not torch.equal(
                full_membrane, reference_membrane
            ):
                raise RuntimeError("full-chain output mismatch")

            timings = {
                "unpacked": median_seconds(unpacked_dense),
                "scalar": median_seconds(scalar_dense),
                "auto": median_seconds(auto_dense),
                "torch": median_seconds(pytorch_dense),
                "packed_full": median_seconds(packed_full),
                "torch_full": median_seconds(pytorch_full),
            }
            print(
                f"density={density:4.0%} | "
                + " | ".join(f"{name}={milliseconds(value)} ms" for name, value in timings.items())
                + f" | dense speedup={timings['torch'] / timings['auto']:.2f}x"
            )

        spikes = (torch.rand(batch, n) < 0.5).to(torch.uint8)
        packed_spikes = pack_spikes(spikes)
        current = (spikes.float() @ float_weights.T).to(torch.int32)
        membrane = torch.zeros(batch, n, dtype=torch.int32)
        primitives = {
            "pack": median_seconds(lambda: pack_spikes(spikes)),
            "unpack": median_seconds(lambda: unpack_spikes(packed_spikes, n)),
            "lif": median_seconds(lambda: lif_step(current, membrane, 16384, 32)),
        }
        print(
            "primitives       | "
            + " | ".join(f"{name}={milliseconds(value)} ms" for name, value in primitives.items())
        )


if __name__ == "__main__":
    main()
