#include <vector>

int array_jump_scan(const std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    int i = 0;
    int visits = 0;
    while (i < n) {
        int jump = arr[i];
        if (jump <= 0) {
            jump = 1;
        }
        i += jump;
        visits++;
    }
    return visits;
}
