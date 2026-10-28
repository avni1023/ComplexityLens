#include <vector>

int find_pair_rare_break(const std::vector<int>& v, int target) {
    int n = static_cast<int>(v.size());
    int ops = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ops++;
            if (v[i] + v[j] == target) {
                return ops;
            }
        }
    }
    return ops;
}
