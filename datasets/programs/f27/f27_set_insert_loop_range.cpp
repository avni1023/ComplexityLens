#include <set>

int set_insert_computed_keys(int n) {
    std::set<int> tree;
    int i = 0;
    while (i < n) {
        int key = (i * 37 + 17) % (n + 1);
        tree.insert(key);
        ++i;
    }
    return tree.empty() ? 0 : *tree.begin();
}
