#include <vector>

// Variant: vector linear scan helper called for each element of outer vector.
// Structurally distinct: operates on vector containers with index iteration.
// n * n = O(n^2).
static long long sum_vector_elements(const std::vector<int>& v) {
    long long s = 0;
    for (std::size_t j = 0; j < v.size(); ++j) {
        s += v[j];
    }
    return s;
}

long long helper_in_loop_vector(const std::vector<int>& v) {
    long long total = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        total += sum_vector_elements(v);
    }
    return total;
}
