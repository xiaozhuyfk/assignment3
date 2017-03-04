#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence) {

    /* 418/618 Students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP. Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode:

       // initialization: see example code above
       score_old[vi] = 1/numNodes;

       while (!converged) {

           // compute score_new[vi] for all nodes vi:
           score_new[vi] = sum over all nodes vj reachable from incoming edges
                              { score_old[vj] / number of edges leaving vj  }
           score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

           score_new[vi] += sum over all nodes vj with no outgoing edges
                              { damping * score_old[vj] / numNodes }

           // compute how much per-node scores have changed
           // quit once algorithm has converged

           global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
           converged = (global_diff < convergence)
       }
    */

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    bool converged = false;

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    std::vector<int> disjoint;
    for (int i = 0; i < numNodes; i++) {
        if (outgoing_size(g, i) == 0) {
            disjoint.push_back(i);
        }
    }

    double *old = (double *) malloc(sizeof(double) * numNodes);
    while (!converged) {
        std::memcpy(old, solution, sizeof(double) * numNodes);

        double disjoint_weight = 0;
        #pragma omp parallel for reduction(+:disjoint_weight)
        for (int j = 0; j < disjoint.size(); j++) {
            disjoint_weight += damping * old[disjoint[j]] / numNodes;
        }

        #pragma omp parallel for
        for (int i = 0; i < numNodes; i++) {
            solution[i] = 0;
            const Vertex* start = incoming_begin(g, i);
            const Vertex* end = incoming_end(g, i);
            double sum = 0;

            for (const Vertex *v = start; v != end; v++) {
                Vertex in = *v;
                sum += old[in] / outgoing_size(g, in);
            }

            solution[i] = (damping * sum) + (1.0 - damping) / numNodes +
                    disjoint_weight;
        }

        double diff = 0;
        #pragma opm parallel for reduction(+:diff)
        for (int i = 0; i < numNodes; i++) {
            diff += std::abs(solution[i] - old[i]);
        }
        converged = (diff < convergence);
    }
}
