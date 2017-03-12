#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>

#include "../include/graph_dist.h"

void compute_disjoint_weight(
        DistGraph &g,
        double damping,
        double *old) {

    int totalVertices = g.total_vertices();

    // Calculate local disjoint weight
    double disjoint_weight = 0.;
    #pragma omp parallel for reduction(+:disjoint_weight)
    for (std::size_t j = 0; j < g.disjoint.size(); j++) {
        disjoint_weight += damping * old[g.disjoint[j]] / totalVertices;
    }

    // gather local disjoint weight to root node
    double *rbuf;
    if (g.world_rank == 0) {
       rbuf = new double[g.world_size * sizeof(double)];
    }
    MPI_Gather(&disjoint_weight, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // sum the distributed disjoint weight
    double total_weight;
    if (g.world_rank == 0) {
        #pragma omp parallel for reduction(+:total_weight)
        for (int mid = 0; mid < g.world_size; mid++) {
            total_weight += rbuf[mid];
        }
    }

    // broadcast total disjoint weight to all nodes
    MPI_Bcast(&total_weight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (g.world_rank == 0) {
        delete(rbuf);
    }

    g.disjoint_weight = total_weight;
}

void compute_global_diff(DistGraph &g, double *solution, double *old) {

    MPI_Request* converge_send_reqs = new MPI_Request[g.world_size];

    // Calculate local convergence
    double diff = 0.;
    #pragma omp parallel for reduction(+:diff)
    for (int i = 0; i < g.vertices_per_process; i++) {
        diff += std::abs(solution[i] - old[i]);
    }

    // Pass local converge score
    double send_value = diff;
    for (int i = 0; i < g.world_size; i++) {
        if (i != g.world_rank) {
            MPI_Isend(&send_value, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &converge_send_reqs[i]);
        }

    }

    //receive and update local converge
    for (int i = 0; i < g.world_size; i++) {
        if (i!=g.world_rank) {
            MPI_Status status;
            double recv_value;
            MPI_Recv(&recv_value, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?
            diff += recv_value;
        }
    }

    delete(converge_send_reqs);

    g.global_diff = diff;
}


void compute_score_test(DistGraph &g, double *solution, double *old, double damping) {
    int total_vertices = g.total_vertices();
    int offset = g.world_rank * g.vertices_per_process;

    std::vector<double*> send_bufs;
    std::vector<int> send_idx;
    std::vector<double*> recv_bufs;
    MPI_Request* send_reqs = new MPI_Request[g.world_size];

    std::map<Vertex, std::vector<double>> buf_map; // buffer to send to other worlds
    std::map<Vertex, double> score_map; // score to update to solution eventually
    //prepare buffer in vector form
    for (int i = 0; i < g.vertices_per_process; i++) {
        double value = old[i] / static_cast<int>(g.outgoing_edges[i].size());

        for (auto &out: g.outgoing_edges[i]){
            int rank = g.get_vertex_owner_rank(out);
            if (rank != g.world_rank){
                //need to send to other world
                int out_offset = rank * g.vertices_per_process;
                buf_map[rank].push_back((out-out_offset)*1.0);
                buf_map[rank].push_back(value);
            }
            else{
                //update local score map on the destination vertex
                score_map[out-offset] += value;
            }
        }
    }

    // initialize buffer size
    // some tips for casting vector to array
    for (int i = 0; i < g.world_size; i++) {
        if (i != g.world_rank) {
            double* send_buf = &buf_map[i][0];
            send_bufs.push_back(send_buf);
            send_idx.push_back(i);
            MPI_Isend(send_buf,
                static_cast<int> (buf_map[i].size()),
                MPI_DOUBLE,
                i, 0, MPI_COMM_WORLD, &send_reqs[i]);
        }
    }

    // Receive and update value
    for (int i = 0; i < g.world_size; i++) {
        if (i!=g.world_rank) {
            MPI_Status status;
            double* recv_buf = new double[g.world_incoming_size[i] * 2];
            recv_bufs.push_back(recv_buf);

            MPI_Recv(recv_buf, g.world_incoming_size[i] * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?

            for(int j = 0; j < g.world_incoming_size[i]; j++) {
                double value = recv_buf[2 * j + 1];
                int recv_vertex = (int) recv_buf[2 * j];
                score_map[recv_vertex] += value;
            }
        }
    }

    // Update the final value to solution to prepare for next iteration
    #pragma omp parallel for
    for (int i = 0; i < g.vertices_per_process ; i++) {
        solution[i] = (damping * score_map[i]) + (1.0 - damping) /  total_vertices + g.disjoint_weight;
    }

    //clear buf
    for (size_t i = 0; i < recv_bufs.size(); i++) {
        delete(recv_bufs[i]);
    }

    delete(send_reqs);
}

inline void compute_score(DistGraph &g, double *solution, double *old, double damping) {

    int vertices_per_process = g.vertices_per_process;
    int total_vertices = g.total_vertices();
    int offset = g.vertices_per_process * g.world_rank;

    std::vector<double *> send_bufs;
    std::vector<double *> recv_bufs;
    MPI_Request* send_reqs = new MPI_Request[g.world_size];

    std::vector<std::vector<double>> buffer_array = std::vector<std::vector<double>>(g.world_size);
    std::map<Vertex, double> score_map;// = std::vector<double>(g.vertices_per_process);

    #pragma omp parallel for
    for (int rank = 0; rank < g.world_size; rank++) {
        buffer_array[rank] = std::vector<double>(g.send_size[rank], 0.0);
    }

    for (auto &e : g.out_edges) {
        int rank = g.get_vertex_owner_rank(e.dest);
        int src = e.src - g.world_rank * g.vertices_per_process;
        double value = old[src] / static_cast<int>(g.outgoing_edges[src].size());
        score_map[e.dest] += value;
    }

    for (int mid = 0; mid < g.world_size; mid++) {
        if (mid != g.world_rank) {
            for (int idx = 0; idx < g.send_mapping[mid].size(); idx++) {
                int dest = g.send_mapping[mid][idx];
                buffer_array[mid][idx] = score_map[dest];
            }

            double* send_buf = &buffer_array[mid][0];
            send_bufs.push_back(send_buf);
            MPI_Isend(send_buf,
                static_cast<int> (buffer_array[mid].size()),
                MPI_DOUBLE,
                mid, 0, MPI_COMM_WORLD, &send_reqs[mid]);
        }
    }

    //prepare buffer in vector form
    /*
    for (int i = 0; i < vertices_per_process; i++) {
        double value = old[i] / static_cast<int>(g.outgoing_edges[i].size());

        for (auto &out : g.outgoing_edges[i]){
            int rank = g.get_vertex_owner_rank(out);
            if (rank != g.world_rank){
                //need to send to other world
                int out_offset = rank * vertices_per_process;
                int index = g.send_mapping[rank][out - out_offset];
                assert(index != -1);
                buffer_array[rank][index] += value;
            } else{
                //update local score map on the destination vertex
                score_map[out - offset] += value;
            }
        }
    }
    */

    // initialize buffer size
    // some tips for casting vector to array
    /*
    for (int i = 0; i < g.world_size; i++) {
        if (i != g.world_rank) {
            double* send_buf = &buffer_array[i][0];
            send_bufs.push_back(send_buf);
            MPI_Isend(send_buf,
                static_cast<int> (buffer_array[i].size()),
                MPI_DOUBLE,
                i, 0, MPI_COMM_WORLD, &send_reqs[i]);
        }
    }
    */

    // Receive and update value
    for (int i = 0; i < g.world_size; i++) {
        if (i != g.world_rank) {
            MPI_Status status;
            double* recv_buf = new double[g.recv_size[i]];
            recv_bufs.push_back(recv_buf);

            MPI_Recv(recv_buf, g.recv_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);

            for(int j = 0; j < g.recv_size[i]; j++) {
                double value = recv_buf[j];
                int recv_vertex = g.recv_mapping[i][j];
                score_map[recv_vertex] += value;
            }
        }
    }

    // Update the final value to solution to prepare for next iteration
    #pragma omp parallel for
    for (int i = 0; i < vertices_per_process ; i++) {
        solution[i] = (damping * score_map[i + g.world_rank * g.vertices_per_process]) + (1.0 - damping) /  total_vertices + g.disjoint_weight;
    }

    //clear buf
    #pragma omp parallel for
    for (size_t i = 0; i < recv_bufs.size(); i++) {
        delete(recv_bufs[i]);
    }

    delete(send_reqs);
}


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
    int vertices_per_process = g.vertices_per_process;
    double equal_prob = 1.0 / totalVertices;

    // initialize per-vertex scores
    #pragma omp parallel for
    for (int i = 0; i < vertices_per_process; ++i) {
        solution[i] = equal_prob;
    }

    double *old = (double *) malloc(sizeof(double) * vertices_per_process);
    int offset = g.world_rank * g.vertices_per_process;

    bool converged = false;
    while (!converged) {
        std::memcpy(old, solution, sizeof(double) * vertices_per_process);

        // Phase 1 : compute global disjoint weight
        compute_disjoint_weight(g, damping, old);

        // Phase 2 : send scores across machine
        /*
        std::vector<double*> send_bufs;
        std::vector<int> send_idx;
        std::vector<double*> recv_bufs;
        MPI_Request* send_reqs = new MPI_Request[g.world_size];

        std::map<Vertex, std::vector<double>> buf_map; // buffer to send to other worlds
        std::map<Vertex, double> score_map; // score to update to solution eventually
        //prepare buffer in vector form
        for (int i = 0; i < vertices_per_process; i++) {
            double value = old[i] / static_cast<int>(g.outgoing_edges[i].size());

            for (auto &out: g.outgoing_edges[i]){
                int rank = g.get_vertex_owner_rank(out);
                if (rank != g.world_rank){
                    //need to send to other world
                    int out_offset = rank * vertices_per_process;
                    buf_map[rank].push_back((out-out_offset)*1.0);
                    buf_map[rank].push_back(value);
                }
                else{
                    //update local score map on the destination vertex
                    score_map[out-offset] += value;
                }
            }
        }

        // initialize buffer size
        // some tips for casting vector to array
        for (int i = 0; i < g.world_size; i++) {
            if (i != g.world_rank) {
                double* send_buf = &buf_map[i][0];
                send_bufs.push_back(send_buf);
                send_idx.push_back(i);
                MPI_Isend(send_buf,
                    static_cast<int> (buf_map[i].size()),
                    MPI_DOUBLE,
                    i, 0, MPI_COMM_WORLD, &send_reqs[i]);
            }
        }

        // Receive and update value
        for (int i = 0; i < g.world_size; i++) {
            if (i!=g.world_rank) {
                MPI_Status status;
                double* recv_buf = new double[g.world_incoming_size[i] * 2];
                recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, g.world_incoming_size[i] * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?

                for(int j = 0; j < g.world_incoming_size[i]; j++) {
                    double value = recv_buf[2 * j + 1];
                    int recv_vertex = (int) recv_buf[2 * j];
                    score_map[recv_vertex] += value;
                }
            }
        }

        // Update the final value to solution to prepare for next iteration
        #pragma omp parallel for
        for (int i = 0; i < vertices_per_process ; i++) {
            solution[i] = (damping * score_map[i]) + (1.0 - damping) /  totalVertices + g.disjoint_weight;
        }

        //clear buf
        for (size_t i = 0; i < recv_bufs.size(); i++) {
            delete(recv_bufs[i]);
        }

        delete(send_reqs);
        */

        compute_score(g, solution, old, damping);
        //printf("%f\n", g.disjoint_weight);

        // Phase 3 : Check for convergence
        compute_global_diff(g, solution, old);

        converged = (g.global_diff < convergence);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(old);
}
