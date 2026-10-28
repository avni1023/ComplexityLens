#include <vector>
#include <cstddef>

std::vector<int> build_prefix_sum_for(const std::vector<int>& v) {
    std::vector<int> pref(v.size(), 0);
    if (v.empty()) return pref;
    pref[0] = v[0];
    for (std::size_t i = 1; i < v.size(); i++) {
        pref[i] = pref[i - 1] + v[i];
    }
    return pref;
}
