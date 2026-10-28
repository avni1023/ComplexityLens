#include <vector>
#include <cstddef>

int get_element_at(const std::vector<int>& v, std::size_t index) {
    if (index < v.size()) {
        return v[index];
    }
    return 0;
}

int fixed_index_helper(const std::vector<int>& v) {
    return get_element_at(v, 0) + get_element_at(v, 5);
}
