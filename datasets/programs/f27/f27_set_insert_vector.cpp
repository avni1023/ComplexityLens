#include <vector>
#include <set>

int set_insert_from_vector(const std::vector<int>& data) {
    std::set<int> unique_elements;
    for (size_t i = 0; i < data.size(); ++i) {
        unique_elements.insert(data[i]);
    }
    return static_cast<int>(unique_elements.size());
}
