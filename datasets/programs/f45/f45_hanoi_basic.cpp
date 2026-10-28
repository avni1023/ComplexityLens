// Seed: classic Tower of Hanoi with three peg characters.
// T(n) = 2T(n-1) + O(1) -> OEXP.
void hanoi_basic(int n, char src, char aux, char tgt) {
    if (n == 0) {
        return;
    }
    hanoi_basic(n - 1, src, tgt, aux);
    // Move top disk from src to tgt (O(1) step).
    (void)src;
    (void)tgt;
    hanoi_basic(n - 1, aux, src, tgt);
}
