// Includes' hash value: 4e050c7ad50d240ab7d5a047be805262

#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&transpose_fp32<
        512, 64, 4
    >);
};
