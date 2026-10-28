#include <vector>

int recursive_split_offset(const std::vector<int>& data, int start, int end) {
    if (start >= end - 1) {
        return 0;
    }
    int mid_probe = start + (end - start) / 2;
    int jump = data[mid_probe];
    int partition = start + (jump % (end - start));
    if (partition <= start || partition >= end) {
        partition = start + 1;
    }
    return data[partition] + recursive_split_offset(data, start, partition) + recursive_split_offset(data, partition, end);
}
