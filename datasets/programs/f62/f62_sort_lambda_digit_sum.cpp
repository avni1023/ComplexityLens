#include <vector>
#include <algorithm>

void sort_by_digit_sum_lambda(std::vector<int>& items) {
    std::sort(items.begin(), items.end(), [](int a, int b) {
        int sum_a = 0;
        int t_a = (a < 0) ? -a : a;
        while (t_a > 0) {
            sum_a += t_a % 10;
            t_a /= 10;
        }
        int sum_b = 0;
        int t_b = (b < 0) ? -b : b;
        while (t_b > 0) {
            sum_b += t_b % 10;
            t_b /= 10;
        }
        return sum_a < sum_b;
    });
}
