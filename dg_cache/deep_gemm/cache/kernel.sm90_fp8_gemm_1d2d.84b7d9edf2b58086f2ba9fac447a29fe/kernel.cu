// Includes' hash value: 1d3fe5c806f820898ddd4baec440d3fb

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K,
        0, 0, 0,
        1,
        16, 64, 128,
        128, 128, 128,
        16,
        128, 128,
        1, false,
        72, GemmType::Normal,
        epilogue::transform::EpilogueIdentity
    >);
};
