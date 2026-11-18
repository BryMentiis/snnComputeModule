import unittest

import torch

from bitsnn_cpu import BitLIFNode, lif_step
from tests import reference


class LIFTests(unittest.TestCase):
    def test_multistep_sequence(self) -> None:
        torch.manual_seed(99)
        inputs = torch.randint(-3, 8, (8, 2, 65), dtype=torch.int32)
        membrane = torch.zeros(2, 65, dtype=torch.int32)
        for step in inputs:
            actual_spikes, actual_membrane = lif_step(step, membrane, 16384, 5)
            expected_spikes, expected_membrane = reference.lif_step(
                step, membrane, 16384, 5
            )
            self.assertTrue(torch.equal(actual_spikes, expected_spikes))
            self.assertTrue(torch.equal(actual_membrane, expected_membrane))
            membrane = actual_membrane

    def test_threshold_and_hard_reset(self) -> None:
        membrane = torch.zeros(1, 1, dtype=torch.int32)
        spikes, membrane = lif_step(torch.full((1, 1), 5, dtype=torch.int32), membrane, 32768, 5)
        self.assertEqual(reference.unpack_bits(spikes, 1).item(), 1)
        self.assertEqual(membrane.item(), 0)

    def test_if_accumulates(self) -> None:
        membrane = torch.zeros(1, 1, dtype=torch.int32)
        spikes, membrane = lif_step(torch.full((1, 1), 2, dtype=torch.int32), membrane, 32768, 5)
        self.assertEqual(reference.unpack_bits(spikes, 1).item(), 0)
        self.assertEqual(membrane.item(), 2)
        spikes, membrane = lif_step(torch.full((1, 1), 3, dtype=torch.int32), membrane, 32768, 5)
        self.assertEqual(reference.unpack_bits(spikes, 1).item(), 1)
        self.assertEqual(membrane.item(), 0)

    def test_negative_leak_truncates_toward_zero(self) -> None:
        membrane = torch.full((1, 1), -5, dtype=torch.int32)
        spikes, membrane = lif_step(torch.zeros(1, 1, dtype=torch.int32), membrane, 16384, 5)
        self.assertEqual(reference.unpack_bits(spikes, 1).item(), 0)
        self.assertEqual(membrane.item(), -2)

    def test_positive_saturation(self) -> None:
        membrane = torch.full((1, 1), torch.iinfo(torch.int32).max, dtype=torch.int32)
        spikes, membrane = lif_step(torch.full((1, 1), 10, dtype=torch.int32), membrane, 32768, torch.iinfo(torch.int32).max)
        self.assertEqual(reference.unpack_bits(spikes, 1).item(), 1)
        self.assertEqual(membrane.item(), 0)

    def test_module_single_and_multi_step_match(self) -> None:
        torch.manual_seed(7)
        inputs = torch.randint(-2, 7, (5, 2, 67), dtype=torch.int32)
        single = BitLIFNode(decay_q=20000, threshold=6)
        multi = BitLIFNode(decay_q=20000, threshold=6, step_mode="m")
        single_outputs = []
        for step in inputs:
            single_outputs.append(single(step))
        multi_outputs = multi(inputs)
        self.assertTrue(torch.equal(torch.stack(single_outputs), multi_outputs))
        self.assertTrue(torch.equal(single.v, multi.v))
        single.reset()
        multi.reset()
        self.assertIsNone(single.v)
        self.assertIsNone(multi.v)


if __name__ == "__main__":
    unittest.main()
