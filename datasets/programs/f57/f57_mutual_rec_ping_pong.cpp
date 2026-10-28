int pong(int n);

int ping(int n) {
    if (n <= 0) {
        return 0;
    }
    return 1 + pong(n - 1);
}

int pong(int n) {
    if (n <= 0) {
        return 0;
    }
    return 2 + ping(n - 1);
}
