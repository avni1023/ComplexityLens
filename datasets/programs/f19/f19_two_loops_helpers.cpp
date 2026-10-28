#include <vector>

int sum_first_half_stride(const std::vector<int>& v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        s += v[i];
    }
    return s;
}

int count_positive(const std::vector<int>& v) {
    int c = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] > 0) {
            ++c;
        }
    }
    return c;
}

int sequential_helpers_driver(const std::vector<int>& data) {
    int s = sum_first_half_stride(data);
    int c = count_positive(data);
    return s + c;
}
