#include <vector>

// Variant: while-loop calling a different declared-only external function transform().
// Structurally distinct: uses while-loop and a vector reference parameter.
// UNKNOWN/external_call: transform() has no visible definition.

// Declaration only — no definition provided.
void transform(const std::vector<int>& v, int n);

int external_call_while_loop(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    int i = 0;
    int ops = 0;
    while (i < n) {
        transform(v, n);
        ++ops;
        ++i;
    }
    return ops;
}
