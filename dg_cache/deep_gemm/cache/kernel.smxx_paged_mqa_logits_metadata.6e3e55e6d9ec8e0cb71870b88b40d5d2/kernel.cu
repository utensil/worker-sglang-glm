// Includes' hash value: ae9a7ebe764dc0a4b1de3de79a816053

#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sched::smxx_paged_mqa_logits_metadata<
        32, 256, 132, false
    >);
};
