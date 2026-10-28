bool is_odd(int n);

bool is_even(int n) {
    if (n == 0) {
        return true;
    }
    return is_odd(n - 1);
}

bool is_odd(int n) {
    if (n == 0) {
        return false;
    }
    return is_even(n - 1);
}
