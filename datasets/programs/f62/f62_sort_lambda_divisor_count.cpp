#include <vector>
#include <algorithm>

void sort_by_divisor_count_lambda(std::vector<int>& data) {
    std::sort(data.begin(), data.end(), [](int a, int b) {
        int divs_a = 0;
        int bound_a = (a < 1) ? 1 : a;
        for (int i = 1; i <= bound_a && i <= 50; ++i) {
            if (bound_a % i == 0) divs_a++;
        }
        int divs_b = 0;
        int bound_b = (b < 1) ? 1 : b;
        for (int j = 1; j <= bound_b && j <= 50; ++j) {
            if (bound_b % j == 0) divs_b++;
        }
        return divs_a < divs_b;
    });
}
