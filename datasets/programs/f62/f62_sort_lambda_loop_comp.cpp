#include <vector>
#include <algorithm>

void sort_by_loop_collatz_weight(std::vector<int>& v) {
    std::sort(v.begin(), v.end(), [](int a, int b) {
        int steps_a = 0;
        int x = a;
        while (x > 1 && steps_a < 20) {
            if (x % 2 == 0) x /= 2;
            else x = 3 * x + 1;
            steps_a++;
        }
        int steps_b = 0;
        int y = b;
        while (y > 1 && steps_b < 20) {
            if (y % 2 == 0) y /= 2;
            else y = 3 * y + 1;
            steps_b++;
        }
        return steps_a < steps_b;
    });
}
