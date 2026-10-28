// Variant: while-loop calling a while-loop helper.
// Structurally distinct: while-loop constructs in both caller and callee.
// n * n = O(n^2).
static int count_linear_while(int n, int val) {
    int count = 0;
    int j = 0;
    while (j < n) {
        if (j == val) {
            ++count;
        }
        ++j;
    }
    return count;
}

int helper_in_loop_while(int n) {
    int total_matches = 0;
    int i = 0;
    while (i < n) {
        total_matches += count_linear_while(n, i);
        ++i;
    }
    return total_matches;
}
