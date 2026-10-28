#include <vector>

// Variant: helper-function split — a helper performs the single disk move;
// the recursive function drives the two sub-problems.
// Structurally distinct: disk-move step is an explicit helper call.
static void move_disk(std::vector<int>& src_peg, std::vector<int>& tgt_peg) {
    tgt_peg.push_back(src_peg.back());
    src_peg.pop_back();
}

void hanoi_helper_swap(int n,
                        std::vector<int>& src,
                        std::vector<int>& aux,
                        std::vector<int>& tgt) {
    if (n == 0) {
        return;
    }
    hanoi_helper_swap(n - 1, src, tgt, aux);
    move_disk(src, tgt);
    hanoi_helper_swap(n - 1, aux, src, tgt);
}
