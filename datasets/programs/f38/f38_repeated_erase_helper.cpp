#include <vector>

int pop_front_element(std::vector<int>& vec) {
    int front_val = vec.front();
    vec.erase(vec.begin());
    return front_val;
}

int drain_vector_helper(std::vector<int>& vec) {
    int checksum = 0;
    while (!vec.empty()) {
        checksum += pop_front_element(vec);
    }
    return checksum;
}
