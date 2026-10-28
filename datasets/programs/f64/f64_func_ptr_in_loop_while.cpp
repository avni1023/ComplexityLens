#include <vector>

// Variant: while-loop invoking a function pointer that transforms and accumulates.
// Structurally distinct: while-loop syntax and accumulator return.
// UNKNOWN / unsupported_construct.
typedef int (*TransformFn)(int);

int func_ptr_in_loop_while(const std::vector<int>& v, TransformFn tf) {
    int sum = 0;
    int i = 0;
    int n = static_cast<int>(v.size());
    while (i < n) {
        sum += tf(v[i]);
        ++i;
    }
    return sum;
}
