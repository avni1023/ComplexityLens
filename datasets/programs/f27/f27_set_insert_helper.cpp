#include <vector>
#include <set>

void insert_element(std::set<int>& s, int value) {
    s.insert(value);
}

int set_insert_via_helper(const std::vector<int>& items) {
    std::set<int> collected;
    for (int val : items) {
        insert_element(collected, val);
    }
    int sum_roots = 0;
    for (int x : collected) {
        sum_roots += x;
    }
    return sum_roots;
}
