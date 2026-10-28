#include <vector>

int extract_bound(const std::vector<int>& items) {
    return items.empty() ? 0 : items[0];
}

int loop_via_extracted_bound(const std::vector<int>& data) {
    int max_steps = extract_bound(data);
    int total = 0;
    for (int step = 0; step < max_steps; ++step) {
        total += step * 2;
    }
    return total;
}
