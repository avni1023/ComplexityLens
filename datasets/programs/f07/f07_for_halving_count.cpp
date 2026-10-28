int count_halvings_for(int n) {
    int steps = 0;
    for (int val = n; val > 1; val /= 2) {
        steps++;
    }
    return steps;
}
