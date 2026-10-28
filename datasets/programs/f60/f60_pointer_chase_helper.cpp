#include <vector>

int step_to_next(const std::vector<int>& links, int current) {
    if (current >= 0 && current < static_cast<int>(links.size())) {
        return links[current];
    }
    return -1;
}

int pointer_chase_helper(const std::vector<int>& links, int start_pos, int finish) {
    int count = 0;
    int node = start_pos;
    while (node != finish && node != -1) {
        node = step_to_next(links, node);
        count++;
    }
    return count;
}
