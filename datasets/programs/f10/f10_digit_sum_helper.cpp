static int extract_digit_sum(int val) {
    int s = 0;
    for (int temp = val; temp > 0; temp /= 10) {
        s += temp % 10;
    }
    return s;
}

int digit_sum(int n) {
    if (n < 0) {
        n = -n;
    }
    return extract_digit_sum(n);
}
