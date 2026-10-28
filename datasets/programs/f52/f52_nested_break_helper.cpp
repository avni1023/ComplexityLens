#include <vector>

bool scan_row_for_rare(const std::vector<int>& row, int threshold) {
    for (size_t col = 0; col < row.size(); ++col) {
        if (row[col] > threshold) {
            return true;
        }
    }
    return false;
}

int matrix_rare_threshold_scan(const std::vector<int>& data, int high_thresh) {
    int n = static_cast<int>(data.size());
    int matched_rows = 0;
    for (int r = 0; r < n; ++r) {
        if (scan_row_for_rare(data, high_thresh)) {
            matched_rows++;
        }
    }
    return matched_rows;
}
