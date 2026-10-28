#include <vector>
#include <algorithm>

// Seed: canonical Floyd-Warshall all-pairs shortest paths algorithm.
// Three nested loops over V vertices: V^3 relaxation operations.
// T(V) = V * V * V * O(1) = Theta(V^3) -> ON3 (with n = V).
void floyd_warshall_basic(std::vector<std::vector<int>>& dist, int V) {
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != -1 && dist[k][j] != -1) {
                    int through_k = dist[i][k] + dist[k][j];
                    if (dist[i][j] == -1 || through_k < dist[i][j]) {
                        dist[i][j] = through_k;
                    }
                }
            }
        }
    }
}
