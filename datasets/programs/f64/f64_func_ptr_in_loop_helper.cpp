#include <vector>

// Variant: helper-function split — a dispatch helper invokes the function pointer.
// Structurally distinct: invocation encapsulated inside helper function.
// UNKNOWN / unsupported_construct.
static int invoke_callback(int val, int (*fn)(int)) {
    return fn(val);
}

int func_ptr_in_loop_helper(const std::vector<int>& v, int (*fn)(int)) {
    int total = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        total += invoke_callback(v[i], fn);
    }
    return total;
}
