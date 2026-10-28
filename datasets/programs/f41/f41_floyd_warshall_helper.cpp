#include <vector>

// Variant: helper-function split — inner row relaxation is extracted to a helper.
// Two outer loops over k and i invoke the helper V^2 times; helper does V work -> O(V^3).
// Structurally distinct: inter-procedural decomposition.
static void relax_row(std::vector<std::vector<int>>& dist, int k, int i, int V) {
    if (dist[i][k] < 0) {
        return;
    }
    for (int j = 0; j < V; ++j) {
        if (dist[k][j] >= 0) {
            int cand = dist[i][k] + dist[k][j];
            if (dist[i][j] < 0 || cand < dist[i][j]) {
                dist[i][j] = cand;
            }
        }
    }
}

void floyd_warshall_helper(std::vector<std::vector<int>>& dist, int V) {
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            relax_row(dist, k, i, V);
        }
    }
}
