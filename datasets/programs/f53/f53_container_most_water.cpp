#include <vector>

static int min_val(int a, int b) {
    return (a < b) ? a : b;
}

int max_water_area(const std::vector<int>& height) {
    int l = 0;
    int r = static_cast<int>(height.size()) - 1;
    int max_area = 0;
    while (l < r) {
        int h = min_val(height[l], height[r]);
        int area = h * (r - l);
        if (area > max_area) {
            max_area = area;
        }
        if (height[l] < height[r]) {
            l++;
        } else {
            r--;
        }
    }
    return max_area;
}
