#include <vector>

// Seed: for-loop with bound v.size() - 2.
// Guards ensure v has at least 2 elements to prevent underflow of size_t.
long long size_minus_literal_for(const std::vector<int>& v) {
    if (v.size() < 2) {
        return 0;
    }
    long long sum = 0;
    for (int i = 0; i < static_cast<int>(v.size()) - 2; ++i) {
        sum += v[i] + v[i + 1] + v[i + 2];
    }
    return sum;
}
