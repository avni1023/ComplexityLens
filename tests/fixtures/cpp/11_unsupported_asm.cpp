// Fixture 11: Inline assembly — an "unsupported construct" for complexity analysis
// libclang parses it as AsmStmt; our analyser must flag it as opaque.
// Expected: AsmStmt node present, complexity analyser emits UNSUPPORTED warning.
int read_tsc() {
    int result;
    __asm__ volatile (
        "rdtsc"
        : "=a"(result)
        :
        : "edx"
    );
    return result;
}
