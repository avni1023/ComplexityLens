// Variant: helper-function split — a visible wrapper helper delegates to the
// declared-only external process(). The wrapper body is visible but process()
// is not; the opaqueness propagates through the helper.
// UNKNOWN/external_call: static analysis cannot resolve process() cost.

// Declaration only — no definition provided.
void process(int n);

static void run_process(int n) {
    process(n);
}

int external_call_helper_wrap(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        run_process(n);
        ++total;
    }
    return total;
}
