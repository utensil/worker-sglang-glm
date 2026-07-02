// Includes' hash value: 07fbd65d478ae156db5fe34542939246

#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mqa_logits<
        32, 128,
        false,
        4, 256,
        3, 3,
        132,
        128, 512,
        float
    >);
};
