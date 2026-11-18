import importlib.metadata

import torch

from bitsnn_cpu import avx2_available, pack_spikes, unpack_spikes
from bitsnn_cpu.networks import BitMLP
from bitsnn_cpu.spikingjelly import reset_net


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    time_steps, batch, n_in, n_hidden, n_out = 4, 2, 65, 67, 61
    spikes = torch.randint(0, 2, (time_steps, batch, n_in), dtype=torch.uint8)
    weight1 = torch.randint(0, 2, (n_hidden, n_in), dtype=torch.uint8)
    weight2 = torch.randint(0, 2, (n_out, n_hidden), dtype=torch.uint8)
    model = BitMLP(weight1, weight2, decay_q=16384, threshold=12)
    packed = pack_spikes(spikes)

    model.reset()
    pytorch_output = model.forward_sequence(packed)

    reset_net(model)
    spikingjelly_output = model.forward_sequence(packed)
    unpacked = unpack_spikes(spikingjelly_output, n_out)

    if not torch.equal(pytorch_output, spikingjelly_output):
        raise RuntimeError("PyTorch and SpikingJelly execution paths differ")
    print(f"PyTorch: {torch.__version__}")
    print(f"SpikingJelly: {importlib.metadata.version('spikingjelly')}")
    print(f"AVX2: {avx2_available()}")
    print(f"Packed output shape: {tuple(spikingjelly_output.shape)}")
    print(f"Unpacked output shape: {tuple(unpacked.shape)}")
    print(f"Output spikes: {int(unpacked.sum())}")
    print("PyTorch and SpikingJelly paths match exactly")


if __name__ == "__main__":
    main()
