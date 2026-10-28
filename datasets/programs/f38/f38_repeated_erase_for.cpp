#include <vector>

void drain_vector_for(std::vector<int>& v) {
    int initial_size = static_cast<int>(v.size());
    for (int i = 0; i < initial_size; ++i) {
        if (!v.empty()) {
            v.erase(v.begin());
        }
    }
}
