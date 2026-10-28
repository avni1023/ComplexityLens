#include <vector>

// Seed: for-loop invoking a function pointer callback for each element.
// Function pointer targets are unknown at compile-time: UNKNOWN / unsupported_construct.
void func_ptr_in_loop_basic(const std::vector<int>& v, void (*fn)(int)) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        fn(v[i]);
    }
}
