import importlib.metadata
import unittest

import torch
from spikingjelly.activation_based import functional

from bitsnn_cpu import pack_spikes
from bitsnn_cpu.networks import BitMLP
from tests import reference


class SpikingJellyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2025)
        self.spikes = torch.randint(0, 2, (5, 2, 65), dtype=torch.uint8)
        self.weight1 = torch.randint(0, 2, (67, 65), dtype=torch.uint8)
        self.weight2 = torch.randint(0, 2, (61, 67), dtype=torch.uint8)
        self.model = BitMLP(
            self.weight1,
            self.weight2,
            decay_q=16384,
            threshold=12,
        )

    def test_mlp_matches_integer_reference(self) -> None:
        packed = pack_spikes(self.spikes)
        actual = self.model.forward_sequence(packed)
        expected, membrane1, membrane2 = reference.mlp(
            self.spikes,
            self.weight1,
            self.weight2,
            decay_q=16384,
            threshold=12,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(self.model.lif1.v, membrane1))
        self.assertTrue(torch.equal(self.model.lif2.v, membrane2))

    def test_spikingjelly_reset_net_restores_initial_behavior(self) -> None:
        packed = pack_spikes(self.spikes)
        first = self.model.forward_sequence(packed)
        self.assertIsNotNone(self.model.lif1.v)
        self.assertIsNotNone(self.model.lif2.v)

        functional.reset_net(self.model)
        self.assertIsNone(self.model.lif1.v)
        self.assertIsNone(self.model.lif2.v)
        second = self.model.forward_sequence(packed)
        self.assertTrue(torch.equal(first, second))

    def test_compatible_version_is_installed(self) -> None:
        self.assertEqual(
            importlib.metadata.version("spikingjelly"),
            "0.0.0.0.14",
        )


if __name__ == "__main__":
    unittest.main()
