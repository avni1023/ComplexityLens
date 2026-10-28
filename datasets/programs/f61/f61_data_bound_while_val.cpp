#include <vector>

int while_bound_by_data_value(const std::vector<int>& v) {
    if (v.empty()) {
        return 0;
    }
    int count = 0;
    int acc = 0;
    while (count < v[0]) {
        acc += (count ^ 3);
        ++count;
    }
    return acc;
}
