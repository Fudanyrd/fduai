//===- fduai/runner/fence.c - Formatting Tensor Data ----------------------===//
//
// To avoid llvm optimizations of tensor computation.
//
//===----------------------------------------------------------------------===//

void m_fence(const float value __attribute__((unused))) {
    // This function is intentionally left empty.
    // It serves as a memory fence to prevent compiler optimizations
    // that could reorder memory operations, ensuring that all previous
    // memory accesses are completed before this point.
    // The value parameter is unused, but it can be used to ensure
    // that the function is not optimized away.

    asm volatile("" ::: "memory");
}
