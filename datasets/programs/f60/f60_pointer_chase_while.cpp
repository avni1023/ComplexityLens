#include <vector>

int pointer_chase_while(const std::vector<int>& next_nodes, int start, int target) {
    int steps = 0;
    int curr = start;
    int n = static_cast<int>(next_nodes.size());
    while (curr >= 0 && curr < n && curr != target) {
        curr = next_nodes[curr];
        steps++;
    }
    return steps;
}
