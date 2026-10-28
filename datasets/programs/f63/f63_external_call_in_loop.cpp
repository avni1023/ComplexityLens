// Seed: a for-loop calling a declared-only external function process(n).
// process() has no visible definition; its complexity is opaque.
// UNKNOWN/external_call: static analysis cannot determine per-call cost.

// Declaration only — no definition provided.
void process(int n);

int external_call_in_loop(int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        process(n);
        ++count;
    }
    return count;
}
