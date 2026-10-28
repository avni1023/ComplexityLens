#include <vector>
#include <cstddef>

long long sum_vector_for(const std::vector<int>& v) {
    long long total = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        total += v[i];
    }
    return total;
}
