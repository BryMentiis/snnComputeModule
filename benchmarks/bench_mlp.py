import torch

from bitsnn_cpu import pack_spikes, unpack_spikes
from bitsnn_cpu.networks import BitMLP
from benchmarks.common import median_seconds, milliseconds


def reference_lif(
    current: torch.Tensor,
    membrane: torch.Tensor,
    decay_q: int,
    threshold: int,
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


def reference_mlp(
    spikes: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    decay_q: int,
    threshold: int,
) -> torch.Tensor:
    membrane1 = torch.zeros(spikes.shape[1], weight1.shape[0], dtype=torch.int64)
    membrane2 = torch.zeros(spikes.shape[1], weight2.shape[0], dtype=torch.int64)
    outputs = []
    for step in spikes:
        current1 = step.to(torch.int64) @ weight1.to(torch.int64).T
        hidden, membrane1 = reference_lif(current1, membrane1, decay_q, threshold)
        current2 = hidden.to(torch.int64) @ weight2.to(torch.int64).T
        output, membrane2 = reference_lif(current2, membrane2, decay_q, threshold)
        outputs.append(output)
    return torch.stack(outputs)


def main() -> None:
    torch.manual_seed(17)
    torch.set_num_threads(1)
    time_steps, batch, n_in, n_hidden, n_out = 16, 8, 256, 512, 128
    decay_q, threshold = 16384, 16
    spikes = (torch.rand(time_steps, batch, n_in) < 0.1).to(torch.uint8)
    weight1 = (torch.rand(n_hidden, n_in) < 0.5).to(torch.uint8)
    weight2 = (torch.rand(n_out, n_hidden) < 0.5).to(torch.uint8)
    packed = pack_spikes(spikes)
    model = BitMLP(weight1, weight2, decay_q, threshold)

    expected = reference_mlp(spikes, weight1, weight2, decay_q, threshold)
    model.reset()
    actual = unpack_spikes(model.forward_sequence(packed), n_out)
    if not torch.equal(actual, expected):
        raise RuntimeError("MLP output mismatch")

    def reference_run() -> torch.Tensor:
        return reference_mlp(spikes, weight1, weight2, decay_q, threshold)

    def scalar_run() -> torch.Tensor:
        model.reset()
        model.fc1.backend = "scalar"
        model.fc2.backend = "scalar"
        return model.forward_sequence(packed)

    def auto_run() -> torch.Tensor:
        model.reset()
        model.fc1.backend = "auto"
        model.fc2.backend = "auto"
        return model.forward_sequence(packed)

    def full_run() -> torch.Tensor:
        model.reset()
        model.fc1.backend = "auto"
        model.fc2.backend = "auto"
        packed_input = pack_spikes(spikes)
        packed_output = model.forward_sequence(packed_input)
        return unpack_spikes(packed_output, n_out)

    timings = {
        "torch_reference": median_seconds(reference_run),
        "packed_scalar": median_seconds(scalar_run),
        "packed_auto": median_seconds(auto_run),
        "packed_with_boundaries": median_seconds(full_run),
    }
    print(
        f"T={time_steps}, B={batch}, sizes={n_in}->{n_hidden}->{n_out}; "
        f"threads={torch.get_num_threads()}"
    )
    for name, value in timings.items():
        print(f"{name:24s}: {milliseconds(value)} ms/sequence")
    print(
        f"End-to-end speedup vs integer PyTorch reference: "
        f"{timings['torch_reference'] / timings['packed_with_boundaries']:.2f}x"
    )
    unpacked_io_bytes = spikes.numel() + time_steps * batch * n_out
    packed_io_bytes = packed.numel() * 8 + time_steps * batch * ((n_out + 63) // 64) * 8
    unpacked_weight_bytes = weight1.numel() + weight2.numel()
    packed_weight_bytes = (
        model.fc1.packed_weight.numel() + model.fc2.packed_weight.numel()
    ) * 8
    membrane_bytes = 2 * batch * (n_hidden + n_out) * 4
    print(f"I/O spike storage: uint8={unpacked_io_bytes} B, packed={packed_io_bytes} B")
    print(f"Weight storage: uint8={unpacked_weight_bytes} B, packed={packed_weight_bytes} B")
    print(f"Two int32 membrane states: {membrane_bytes} B")


if __name__ == "__main__":
    main()
