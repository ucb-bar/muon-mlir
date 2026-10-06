#include <mu_intrinsics.h>
#include <stdint.h>

// These wrappers keep the compiler's runtime ABI separate from the source
// spelling of the hardware intrinsics. Compile with the same flags and stack
// stride as libmuonrt.a and every emitted Muon object.
extern "C" void muon_mlir_barrier(uint32_t id, uint32_t warps) {
  mu_barrier(id, warps);
}
extern "C" void muon_mlir_fence() { mu_fence(); }
extern "C" void muon_mlir_smem_fence() { mu_fence_smem(); }
