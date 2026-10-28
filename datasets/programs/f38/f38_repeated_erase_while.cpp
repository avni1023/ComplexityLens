#include <vector>

int drain_vector_while(std::vector<int>& v) {
    int removed_count = 0;
    while (!v.empty()) {
        v.erase(v.begin());
        removed_count++;
    }
    return removed_count;
}
