int halve_value(int x) {
    return x >> 1;
}

int count_halvings_helper(int n) {
    int count = 0;
    int current = n;
    while (current > 1) {
        current = halve_value(current);
        count++;
    }
    return count;
}
