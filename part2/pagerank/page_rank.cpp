#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../include/graph_dist.h"

typedef struct edge_package edge_package;

struct edge_package {
    int recv_vertex;
    double score;
};

/*
 * pageRank--
 *
 * Computes page rank on a distributed graph g
 *
 * Per-vertex scores for all vertices *owned by this node* (not all
 * vertices in the graph) should be placed in `solution` upon
 * completion.
 */
void pageRank(DistGraph &g, double* solution, double damping, double convergence) {

    // TODO FOR 15-418/618 STUDENTS:

    // Implement the distributed page rank algorithm here. This is
    // very similar to what you implemnted in Part 1, except in this
    // case, the graph g is distributed across cluster nodes.

    // Each node in the cluster is only aware of the outgoing edge
    // topology for the vertices it "owns".  The cluster nodes will
    // need to coordinate to determine what information.

    // note: we give you starter code below to initialize scores for
    // ALL VERTICES in the graph, but feel free to modify as desired.
    // Keep in mind the `solution` array returned to the caller should
    // only have scores for the local vertices

    int totalVertices = g.total_vertices();
    double equal_prob = 1.0/totalVertices;
    int local_size = g.end_vertex-g.start_vertex+1;
    int vertices_per_process = g.vertices_per_process; //numNodes
    std::vector<int> disjoint = g.disjoint;
    bool converged = false;

    // initialize per-vertex scores
    #pragma omp parallel for
    for (int i = 0; i < vertices_per_process; ++i) {
        solution[i] = equal_prob;
    }

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    double *old = (double *) malloc(sizeof(double) * vertices_per_process);

    double* disjoint_send_buf = new double[1];
    double* disjoint_recv_buf = new double[1];
    MPI_Request* disjoint_send_reqs = new MPI_Request[g.world_size];

    double* converge_send_buf = new double[1];
    double* converge_recv_buf = new double[1];
    MPI_Request* converge_send_reqs = new MPI_Request[g.world_size];

    MPI_Request* send_reqs = new MPI_Request[g.world_size];

    while (!converged) {
        std::memcpy(old, solution, sizeof(double) * vertices_per_process);

        // Phase 1 : update disjoint weight
        // Calculate local disjoint weight
        
        double disjoint_weight = 0.;
        #pragma omp parallel for reduction(+:disjoint_weight)
        for (std::size_t j = 0; j < disjoint.size(); j++) {
            disjoint_weight += damping * old[disjoint[j]] / totalVertices;
        }

        if (g.world_rank) {
            disjoint_send_buf[0] = disjoint_weight;
            MPI_Isend(disjoint_send_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &disjoint_send_reqs[0]);

            MPI_Status status;
            MPI_Recv(disjoint_recv_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            disjoint_weight =  disjoint_recv_buf[0];
        } else {
            for (int i = 1; i < g.world_size; i++) {
                MPI_Status status;
                MPI_Recv(disjoint_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                disjoint_weight += disjoint_recv_buf[0];
            }
            disjoint_send_buf[0] = disjoint_weight;
            for (int i = 1; i < g.world_size; i++) {
                MPI_Isend(disjoint_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &disjoint_send_reqs[i]);
            }
        }
        
        // Phase 2 : send scores across machine

        std::vector<double*> send_bufs;
        std::vector<double*> recv_bufs;
        std::vector<int> send_idx;

        // Calculate local score to send to other worlds
        std::vector<double> local_outedge_score = std::vector<double>(local_size);

        // Calculate updated score every vertex should receive from this world
        std::vector<double> vtx_score = std::vector<double>(g.outgoing_edge_gather.size());

        // local world score eventually to be updated in <solution>
        std::vector<double> local_score =  std::vector<double>(local_size);

        // Calculate local outgoing edges score
        for (int i = 0; i < local_size; i++) {
            if (g.outgoing_edges[i].size()) {
                local_outedge_score[i] = old[i] / static_cast<int>(g.outgoing_edges[i].size());
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < g.outgoing_edge_gather.size(); i++) {
            for (auto& local_v: g.outgoing_edge_gather[i]) {
                vtx_score[i] += local_outedge_score[local_v];
            }
        }

        // Send score to other worlds
        for (int r = 0; r < g.world_size; r++) {

            int send_buf_size = static_cast<int> (g.rank_outedge_lookup[r].size());
            double* send_buf = new double[send_buf_size];
            if (send_buf_size) {
                send_bufs.push_back(send_buf);
                send_idx.push_back(r);
            }

            for (int i = 0; i < send_buf_size; i++) {
                send_buf[i] = vtx_score[g.rank_outedge_lookup[r][i]];
            }

            if (r != g.world_rank && send_buf_size) {
                // We should send this buf
                MPI_Isend(send_buf, send_buf_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &send_reqs[r]);
            } else if (r == g.world_rank) {
                // We should internalize this buf

                for (int i = 0; i < send_buf_size; i++) {
                    local_score[g.rank_inedge_lookup[r][i]] += send_buf[i];
                }
            }
        }

        // Receive and update from other worlds
        for (int r = 0; r < g.world_size; r++) {
            int recv_buf_size = static_cast<int> (g.rank_inedge_lookup[r].size());
            if (r != g.world_rank && recv_buf_size) {
                MPI_Status status;
                double* recv_buf = new double[recv_buf_size];
                recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, recv_buf_size, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &status);

                for (int i = 0; i < recv_buf_size; i++) {
                    local_score[g.rank_inedge_lookup[r][i]] += recv_buf[i];
                }
            }
        }

        // Update the final value to solution to prepare for next iteration
        for (int i = 0; i < local_size ; i++) {
            solution[i] = (damping * local_score[i]) + (1.0 - damping) /  totalVertices + disjoint_weight;
        }


        for (size_t i = 0; i < send_bufs.size(); i++) {
            if (send_idx[i] != g.world_rank) {
                MPI_Status status;
                MPI_Wait(&send_reqs[send_idx[i]], &status);
            }
            delete(send_bufs[i]);
        }


        //clear buf
        for (size_t i = 0; i < recv_bufs.size(); i++) {
            delete(recv_bufs[i]);
        }

        // Phase 3 : Check for convergence

        // Calculate local convergence
        double diff = 0.;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < vertices_per_process; i++) {
            diff += std::abs(solution[i] - old[i]);
        }

        if (g.world_rank) {
            converge_send_buf[0] = diff;
            MPI_Isend(converge_send_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &converge_send_reqs[0]);

            MPI_Status status;
            MPI_Recv(converge_recv_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            diff = converge_recv_buf[0];
        } else {
            for (int i = 1; i < g.world_size; i++) {
                MPI_Status status;
                MPI_Recv(converge_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                diff += converge_recv_buf[0];
            }
            converge_send_buf[0] = diff;
            for (int i = 1; i < g.world_size; i++) {
                MPI_Isend(converge_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &converge_send_reqs[i]);
            }
        }

        //Make sure all the sends are received
        MPI_Barrier(MPI_COMM_WORLD);

        converged = (diff < convergence);
    }
    // clear disjoint buf
    delete(disjoint_send_buf);
    delete(disjoint_recv_buf);
    delete(disjoint_send_reqs);

    // clear disjoint buf
    delete(converge_send_buf);
    delete(converge_recv_buf);
    delete(converge_send_reqs);

    delete(send_reqs);

    free(old);
}
