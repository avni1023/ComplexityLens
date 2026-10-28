#include <vector>

static int get_boundary_element(const std::vector<int>& v, bool first) {
    return first ? v[0] : v[v.size() - 1];
}

int boundary_sum(const std::vector<int>& v) {
    if (v.size() == 0) {
        return -1;
    }
    return get_boundary_element(v, true) + get_boundary_element(v, false);
}
