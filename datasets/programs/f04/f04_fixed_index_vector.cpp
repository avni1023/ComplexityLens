#include <vector>

int fixed_index_vector(const std::vector<int>& a) {
    if (a.size() > 5) {
        return a[0] + a[5];
    }
    return 0;
}
