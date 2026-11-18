#include "bit_kernels.h"

#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

using at::Tensor;

void check_cpu_contiguous(const Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_binary_tensor(const Tensor& tensor, const char* name) {
    check_cpu_contiguous(tensor, name);
    TORCH_CHECK(
        tensor.scalar_type() == at::kByte || tensor.scalar_type() == at::kBool,
        name,
        " must have dtype torch.uint8 or torch.bool");
}

void check_packed_tensor(const Tensor& tensor, const char* name) {
    check_cpu_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must have dtype torch.int64");
    TORCH_CHECK(tensor.dim() >= 1, name, " must have at least one dimension");
}

void check_int32_tensor(const Tensor& tensor, const char* name) {
    check_cpu_contiguous(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype torch.int32");
}

const uint8_t* binary_data(const Tensor& tensor) {
    if (tensor.scalar_type() == at::kByte) {
        return tensor.data_ptr<uint8_t>();
    }
    return reinterpret_cast<const uint8_t*>(tensor.data_ptr<bool>());
}

Tensor pack_tensor(const Tensor& input) {
    check_binary_tensor(input, "input");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    const int64_t n = input.size(-1);
    TORCH_CHECK(n > 0, "the packed dimension must be non-empty");
    const int64_t words = (n + 63) / 64;
    const int64_t rows = input.numel() / n;

    std::vector<int64_t> output_sizes = input.sizes().vec();
    output_sizes.back() = words;
    Tensor output = at::empty(output_sizes, input.options().dtype(at::kLong));
    const bool valid = bitsnn_cpu::pack_bits(
        binary_data(input),
        rows,
        n,
        reinterpret_cast<uint64_t*>(output.data_ptr<int64_t>()));
    TORCH_CHECK(valid, "input values must be 0 or 1");
    return output;
}

Tensor pack_spikes(const Tensor& spikes) {
    return pack_tensor(spikes);
}

Tensor pack_weights(const Tensor& weights) {
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [Nout, Nin]");
    return pack_tensor(weights);
}

Tensor unpack_spikes(const Tensor& packed, int64_t n) {
    check_packed_tensor(packed, "packed");
    TORCH_CHECK(n > 0, "n must be positive");
    const int64_t words = (n + 63) / 64;
    TORCH_CHECK(packed.size(-1) == words, "packed last dimension does not match n");
    const int64_t rows = packed.numel() / words;

    std::vector<int64_t> output_sizes = packed.sizes().vec();
    output_sizes.back() = n;
    Tensor output = at::empty(output_sizes, packed.options().dtype(at::kByte));
    bitsnn_cpu::unpack_bits(
        reinterpret_cast<const uint64_t*>(packed.data_ptr<int64_t>()),
        rows,
        n,
        output.data_ptr<uint8_t>());
    return output;
}

enum class DenseBackend {
    automatic,
    scalar,
    avx2,
};

Tensor binary_dense_impl(
    const Tensor& spikes,
    const Tensor& weights,
    int64_t n_in,
    DenseBackend backend) {
    check_packed_tensor(spikes, "spikes");
    check_packed_tensor(weights, "weights");
    TORCH_CHECK(spikes.dim() == 2, "spikes must have shape [B, ceil(Nin/64)]");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [Nout, ceil(Nin/64)]");
    TORCH_CHECK(n_in > 0 && n_in <= std::numeric_limits<int32_t>::max(), "invalid n_in");
    const int64_t words = (n_in + 63) / 64;
    TORCH_CHECK(spikes.size(1) == words, "spikes last dimension does not match n_in");
    TORCH_CHECK(weights.size(1) == words, "weights last dimension does not match n_in");

    const int64_t batch = spikes.size(0);
    const int64_t n_out = weights.size(0);
    Tensor output = at::empty({batch, n_out}, spikes.options().dtype(at::kInt));
    const uint64_t* spike_data =
        reinterpret_cast<const uint64_t*>(spikes.data_ptr<int64_t>());
    const uint64_t* weight_data =
        reinterpret_cast<const uint64_t*>(weights.data_ptr<int64_t>());
    int32_t* output_data = output.data_ptr<int32_t>();

    using DotFunction = int32_t (*)(const uint64_t*, const uint64_t*, int64_t);
    DotFunction dot = &bitsnn_cpu::binary_dot_scalar;
    if (backend == DenseBackend::automatic && bitsnn_cpu::avx2_available()) {
        dot = &bitsnn_cpu::binary_dot_avx2;
    } else if (backend == DenseBackend::avx2) {
        TORCH_CHECK(bitsnn_cpu::avx2_available(), "AVX2 is not available on this CPU");
        dot = &bitsnn_cpu::binary_dot_avx2;
    }
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t output_index = 0; output_index < n_out; ++output_index) {
            const uint64_t* spike_row = spike_data + batch_index * words;
            const uint64_t* weight_row = weight_data + output_index * words;
            output_data[batch_index * n_out + output_index] = dot(spike_row, weight_row, n_in);
        }
    }
    return output;
}

Tensor binary_dense(const Tensor& spikes, const Tensor& weights, int64_t n_in) {
    return binary_dense_impl(spikes, weights, n_in, DenseBackend::automatic);
}

Tensor binary_dense_scalar(const Tensor& spikes, const Tensor& weights, int64_t n_in) {
    return binary_dense_impl(spikes, weights, n_in, DenseBackend::scalar);
}

Tensor binary_dense_avx2(const Tensor& spikes, const Tensor& weights, int64_t n_in) {
    return binary_dense_impl(spikes, weights, n_in, DenseBackend::avx2);
}

Tensor binary_dense_unpacked(const Tensor& spikes, const Tensor& weights) {
    check_binary_tensor(spikes, "spikes");
    check_binary_tensor(weights, "weights");
    TORCH_CHECK(spikes.dim() == 2, "spikes must have shape [B, Nin]");
    TORCH_CHECK(weights.dim() == 2, "weights must have shape [Nout, Nin]");
    TORCH_CHECK(spikes.size(1) == weights.size(1), "Nin dimensions do not match");
    const int64_t n_in = spikes.size(1);
    TORCH_CHECK(n_in > 0 && n_in <= std::numeric_limits<int32_t>::max(), "invalid Nin");
    const int64_t batch = spikes.size(0);
    const int64_t n_out = weights.size(0);
    Tensor output = at::empty({batch, n_out}, spikes.options().dtype(at::kInt));

    const uint8_t* spike_data = binary_data(spikes);
    const uint8_t* weight_data = binary_data(weights);
    int32_t* output_data = output.data_ptr<int32_t>();
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t output_index = 0; output_index < n_out; ++output_index) {
            output_data[batch_index * n_out + output_index] = bitsnn_cpu::binary_dot_unpacked(
                spike_data + batch_index * n_in,
                weight_data + output_index * n_in,
                n_in);
        }
    }
    return output;
}

std::tuple<Tensor, Tensor> lif_step(
    const Tensor& input,
    const Tensor& membrane,
    int64_t decay_q,
    int64_t threshold,
    int64_t v_reset) {
    check_int32_tensor(input, "input");
    check_int32_tensor(membrane, "membrane");
    TORCH_CHECK(input.dim() == 2, "input must have shape [B, N]");
    TORCH_CHECK(input.sizes() == membrane.sizes(), "input and membrane shapes must match");
    TORCH_CHECK(decay_q >= 0 && decay_q <= bitsnn_cpu::kQ15One, "decay_q must be in [0, 32768]");
    TORCH_CHECK(
        threshold >= std::numeric_limits<int32_t>::min() &&
            threshold <= std::numeric_limits<int32_t>::max(),
        "threshold must fit int32");
    TORCH_CHECK(
        v_reset >= std::numeric_limits<int32_t>::min() &&
            v_reset <= std::numeric_limits<int32_t>::max(),
        "v_reset must fit int32");

    const int64_t rows = input.size(0);
    const int64_t n = input.size(1);
    TORCH_CHECK(n > 0, "the neuron dimension must be non-empty");
    const int64_t words = (n + 63) / 64;
    Tensor spikes = at::empty({rows, words}, input.options().dtype(at::kLong));
    Tensor next_membrane = at::empty_like(membrane);
    bitsnn_cpu::lif_step(
        input.data_ptr<int32_t>(),
        membrane.data_ptr<int32_t>(),
        rows,
        n,
        static_cast<int32_t>(decay_q),
        static_cast<int32_t>(threshold),
        static_cast<int32_t>(v_reset),
        reinterpret_cast<uint64_t*>(spikes.data_ptr<int64_t>()),
        next_membrane.data_ptr<int32_t>());
    return {spikes, next_membrane};
}

bool avx2_available() {
    return bitsnn_cpu::avx2_available();
}

}  // namespace

TORCH_LIBRARY(bitsnn_cpu, m) {
    m.def("pack_spikes(Tensor spikes) -> Tensor");
    m.def("pack_weights(Tensor weights) -> Tensor");
    m.def("unpack_spikes(Tensor packed, int n) -> Tensor");
    m.def("binary_dense(Tensor spikes, Tensor weights, int n_in) -> Tensor");
    m.def("binary_dense_scalar(Tensor spikes, Tensor weights, int n_in) -> Tensor");
    m.def("binary_dense_avx2(Tensor spikes, Tensor weights, int n_in) -> Tensor");
    m.def("binary_dense_unpacked(Tensor spikes, Tensor weights) -> Tensor");
    m.def(
        "lif_step(Tensor input, Tensor membrane, int decay_q, int threshold, int v_reset=0) -> (Tensor, Tensor)");
    m.def("avx2_available() -> bool");
}

TORCH_LIBRARY_IMPL(bitsnn_cpu, CPU, m) {
    m.impl("pack_spikes", &pack_spikes);
    m.impl("pack_weights", &pack_weights);
    m.impl("unpack_spikes", &unpack_spikes);
    m.impl("binary_dense", &binary_dense);
    m.impl("binary_dense_scalar", &binary_dense_scalar);
    m.impl("binary_dense_avx2", &binary_dense_avx2);
    m.impl("binary_dense_unpacked", &binary_dense_unpacked);
    m.impl("lif_step", &lif_step);
    m.impl("avx2_available", &avx2_available);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avx2_available", &avx2_available);
}
