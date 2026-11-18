import unittest

import torch

from bitsnn_cpu import (
    avx2_available,
    binary_dense,
    binary_dense_unpacked,
    pack_spikes,
    pack_weights,
    unpack_spikes,
)
from tests import reference


class BitPackingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def test_round_trip_and_reference(self) -> None:
        for n in (1, 63, 64, 65, 127, 128):
            for pattern in ("zero", "one", "random"):
                with self.subTest(n=n, pattern=pattern):
                    if pattern == "zero":
                        spikes = torch.zeros(3, 2, n, dtype=torch.uint8)
                    elif pattern == "one":
                        spikes = torch.ones(3, 2, n, dtype=torch.uint8)
                    else:
                        spikes = torch.randint(0, 2, (3, 2, n), dtype=torch.uint8)
                    packed = pack_spikes(spikes)
                    self.assertEqual(packed.dtype, torch.int64)
                    self.assertEqual(packed.shape, (3, 2, (n + 63) // 64))
                    self.assertTrue(torch.equal(packed, reference.pack_bits(spikes)))
                    self.assertTrue(torch.equal(unpack_spikes(packed, n), spikes))

    def test_bool_and_direct_torch_op(self) -> None:
        spikes = torch.randint(0, 2, (2, 65), dtype=torch.uint8).bool()
        packed = torch.ops.bitsnn_cpu.pack_spikes(spikes)
        self.assertTrue(torch.equal(packed, reference.pack_bits(spikes)))
        self.assertTrue(torch.equal(unpack_spikes(packed, 65), spikes.uint8()))

    def test_noncontiguous_input(self) -> None:
        source = torch.randint(0, 2, (2, 130), dtype=torch.uint8)
        spikes = source[:, ::2]
        self.assertFalse(spikes.is_contiguous())
        self.assertTrue(torch.equal(pack_spikes(spikes), reference.pack_bits(spikes)))

    def test_invalid_values(self) -> None:
        spikes = torch.tensor([[0, 2, 1]], dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            pack_spikes(spikes)


class BinaryDenseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4321)

    def test_dense_against_references(self) -> None:
        for n in (1, 63, 64, 65, 127, 128, 256):
            with self.subTest(n=n):
                spikes = torch.randint(0, 2, (3, n), dtype=torch.uint8)
                weights = torch.randint(0, 2, (5, n), dtype=torch.uint8)
                packed_spikes = pack_spikes(spikes)
                packed_weights = pack_weights(weights)
                expected = reference.dense(spikes, weights)
                self.assertTrue(
                    torch.equal(binary_dense_unpacked(spikes, weights), expected)
                )
                self.assertTrue(
                    torch.equal(
                        binary_dense(packed_spikes, packed_weights, n, "scalar"),
                        expected,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        binary_dense(packed_spikes, packed_weights, n), expected
                    )
                )
                if avx2_available():
                    self.assertTrue(
                        torch.equal(
                            binary_dense(packed_spikes, packed_weights, n, "avx2"),
                            expected,
                        )
                    )

    def test_all_ones_and_tail_mask(self) -> None:
        for n in (1, 64, 65, 128):
            spikes = torch.ones(1, n, dtype=torch.uint8)
            weights = torch.ones(n, n, dtype=torch.uint8)
            output = binary_dense(pack_spikes(spikes), pack_weights(weights), n)
            self.assertTrue(torch.equal(output, torch.full((1, n), n, dtype=torch.int32)))

    def test_avx2_is_detected_on_target(self) -> None:
        self.assertTrue(avx2_available())


if __name__ == "__main__":
    unittest.main()
