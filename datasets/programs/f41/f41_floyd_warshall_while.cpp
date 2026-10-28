#include <vector>

// Variant: Floyd-Warshall expressed with while loops.
// Structurally distinct: manual increment while loops.
// T(V) = V^3 operations -> ON3.
void floyd_warshall_while(std::vector<std::vector<int>>& dist, int V) {
    int k = 0;
    while (k < V) {
        int i = 0;
        while (i < V) {
            int j = 0;
            while (j < V) {
                if (dist[i][k] >= 0 && dist[k][j] >= 0) {
                    int cand = dist[i][k] + dist[k][j];
                    if (dist[i][j] < 0 || cand < dist[i][j]) {
                        dist[i][j] = cand;
                    }
                }
                ++j;
            }
            ++i;
        }
        ++k;
    }
}
