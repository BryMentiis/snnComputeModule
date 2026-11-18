import torch
from torch import nn

from .modules import BitLIFNode, PackedBitLinear
from .ops import DenseBackend


class BitMLP(nn.Module):
    def __init__(
        self,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        decay_q: int,
        threshold: int,
        v_reset: int = 0,
        backend: DenseBackend = "auto",
    ) -> None:
        super().__init__()
        if weight1.ndim != 2 or weight2.ndim != 2:
            raise ValueError("weights must be rank-2 tensors")
        if weight1.shape[0] != weight2.shape[1]:
            raise ValueError("hidden dimensions do not match")
        self.n_in = weight1.shape[1]
        self.n_hidden = weight1.shape[0]
        self.n_out = weight2.shape[0]
        self.fc1 = PackedBitLinear.from_binary(weight1, backend)
        self.lif1 = BitLIFNode(decay_q, threshold, v_reset)
        self.fc2 = PackedBitLinear.from_binary(weight2, backend)
        self.lif2 = BitLIFNode(decay_q, threshold, v_reset)

    def reset(self) -> None:
        self.lif1.reset()
        self.lif2.reset()

    def forward(self, packed_spikes: torch.Tensor) -> torch.Tensor:
        hidden = self.lif1(self.fc1(packed_spikes))
        return self.lif2(self.fc2(hidden))

    def forward_sequence(self, packed_spikes: torch.Tensor) -> torch.Tensor:
        if packed_spikes.ndim != 3:
            raise ValueError("sequence input must have shape [T, B, ceil(N/64)]")
        return torch.stack([self(step) for step in packed_spikes], dim=0)
