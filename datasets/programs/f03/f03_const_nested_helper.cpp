int helper_row(int row_idx) {
    int row_sum = 0;
    for (int col = 0; col < 10; ++col) {
        row_sum += row_idx ^ col;
    }
    return row_sum;
}

int nested_helper_const() {
    int grand_total = 0;
    for (int row = 0; row < 10; ++row) {
        grand_total += helper_row(row);
    }
    return grand_total;
}
