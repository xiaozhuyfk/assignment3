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

    int vertices_per_process = g.vertices_per_process; //numNodes

    std::vector<double> score_curr(vertices_per_process);
    std::vector<double> score_next(g.vertices_per_process);

    // initialize per-vertex scores
    #pragma omp parallel for
    for (Vertex i = 0; i < vertices_per_process; i++) {
        score_curr[i] = equal_prob; // push local vertex index
    }

    bool converged = false;
    //initialize local disjoint set
    std::vector<int> disjoint;
    for (int v = g.start_vertex; v < g.end_vertex+1; v++) { //inclusive?
        if (g.not_disjoint.find(v)==g.not_disjoint.end()) {
            disjoint.push_back(v); //push global vertex index
        }
    }

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    double *old = (double *) malloc(sizeof(double) * vertices_per_process);
    int offset = g.world_rank * g.vertices_per_process;
    int offset_bit = (int) ceil(log10(vertices_per_process));

    while (!converged) {
        std::memcpy(old, solution, sizeof(double) * vertices_per_process);

        std::vector<double*> disjoint_send_bufs;
        std::vector<int> disjoint_send_idx;
        std::vector<double*> disjoint_recv_bufs;

        MPI_Request* disjoint_send_reqs = new MPI_Request[g.world_size];

        //MPI_Status* probe_status = new MPI_Status[g.world_size];

        // Phase 1 : update disjoint weight 
        // Calculate local disjoint weight
        double disjoint_weight = 0.;
        #pragma omp parallel for reduction(+:disjoint_weight) 
        for (std::size_t j = 0; j < disjoint.size(); j++) {
            disjoint_weight += damping * old[disjoint[j]] / totalVertices;
        }
        //pass local disjoint 
        double* disjoint_send_buf = new double[1];
        
        for (int i = 0; i < g.world_size; i++) {
            if (i != g.world_rank) {
                disjoint_send_bufs.push_back(disjoint_send_buf);
                disjoint_send_idx.push_back(i);
                disjoint_send_buf[0] = disjoint_weight;
                MPI_Isend(disjoint_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &disjoint_send_reqs[i]);
            }
            
        }
        //receive and update local disjoint
        for (int i = 0; i < g.world_size; i++) {
            if (i!=g.world_rank) {
                MPI_Status status;
                double* recv_buf = new double[1];
                disjoint_recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?
                disjoint_weight += recv_buf[0];
            }
        }


        //clear disjoint buf
        for (size_t i = 0; i < disjoint_send_bufs.size(); i++) {
            MPI_Status status;
            MPI_Wait(&disjoint_send_reqs[disjoint_send_idx[i]], &status);
            delete(disjoint_send_bufs[i]);
        }

        for (size_t i = 0; i < disjoint_recv_bufs.size(); i++) {
            delete(disjoint_recv_bufs[i]);
        }

        delete(disjoint_send_reqs);

        // Phase 2 : Update common vertices
        // initialize buffer for main routine

        std::vector<double*> send_bufs;
        std::vector<int> send_idx;
        std::vector<double*> recv_bufs;
        MPI_Request* send_reqs = new MPI_Request[g.world_size];

        std::map<Vertex, std::vector<double>> buf_map; // buffer to send to other worlds
        std::map<Vertex, double> score_map; // score to update to solution eventually

        //prepare buffer in vector form
        #pragma omp parallel for
        for (int i = 0; i < vertices_per_process; i++) {
            double value = old[i] / static_cast<int>(g.outgoing_edges[i].size());
            int integer_value = (int) value;
            double decimal_value = value-(int) value;

            for (auto &out: g.outgoing_edges[i]){
                int rank = g.get_vertex_owner_rank(out);
                if (rank != g.world_rank){
                    //need to send to other world
                    int out_offset = rank * vertices_per_process;
                    buf_map[rank].push_back(integer_value * offset_bit + (out-out_offset) + decimal_value);
                }
                else{
                    //update local score map on the destination vertex
                    score_map[out-offset] += value;
                }
            }
        }

        // initialize buffer size 
        // some tips for casting vector to array
        // double arr[100];
        // std::copy(v.begin(), v.end(), arr);
        for (int i = 0; i < g.world_size; i++) {
            if (i != g.world_rank) {
                double* send_buf = &buf_map[i][0];  //Need to check if RPC'd; new double[g.outgoing_edges[i]];
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
                double* recv_buf = new double[g.world_incoming_size[i]];
                recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, g.world_incoming_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?
                #pragma omp parallel for
                for(int j = 0; j < g.world_incoming_size[i]; j++) {
                    double value = recv_buf[j];
                    int integer_value = (int) value / offset_bit;
                    int recv_vertex = (int) value - integer_value; // the index recv'd is local
                    double decimal_value = value-(int) value;
                    double final_value = integer_value + decimal_value;

                    score_map[recv_vertex] += final_value;
                }
            }
        }

        // Update the final value to solution to prepare for next iteration
        #pragma omp parallel for
        for (int i = 0; i < vertices_per_process ; i++) {
            solution[i] = (damping * score_map[i]) + (1.0 - damping) /  g.num_vertices + disjoint_weight;
        }

        //clear buf
        for (size_t i = 0; i < send_bufs.size(); i++) {
            MPI_Status status;
            MPI_Wait(&send_reqs[send_idx[i]], &status);
            delete(send_bufs[i]);
        }

        for (size_t i = 0; i < recv_bufs.size(); i++) {
            delete(recv_bufs[i]);
        }

        delete(send_reqs);

        // Phase 3 : Check for convergence
        std::vector<double*> converge_send_bufs;
        std::vector<int> converge_send_idx;
        std::vector<double*> converge_recv_bufs;

        MPI_Request* converge_send_reqs = new MPI_Request[g.world_size];

        // Calculate local convergence
        double diff = 0.;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < vertices_per_process; i++) {
            diff += std::abs(solution[i] - old[i]);
        }
        
        // Pass local converge score
        double* converge_send_buf = new double[1];
        for (int i = 0; i < g.world_size; i++) {
            if (i != g.world_rank) {
                converge_send_bufs.push_back(converge_send_buf);
                converge_send_idx.push_back(i);
                converge_send_buf[0] = diff;
                MPI_Isend(converge_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &converge_send_reqs[i]);
            }
            
        }
        //receive and update local converge
        for (int i = 0; i < g.world_size; i++) {
            if (i!=g.world_rank) {
                MPI_Status status;
                double* recv_buf = new double[1];
                converge_recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?
                diff += recv_buf[0];
            }
        }

        converged = (diff < convergence);

        //clear converge buf
        for (size_t i = 0; i < converge_send_bufs.size(); i++) {
            MPI_Status status;
            MPI_Wait(&converge_send_reqs[converge_send_idx[i]], &status);
            delete(converge_send_bufs[i]);
        }

        for (size_t i = 0; i < converge_recv_bufs.size(); i++) {
            delete(converge_recv_bufs[i]);
        }

        delete(converge_send_reqs);
    
    }

    free(old);

    /*

      Repeating basic pagerank pseudocode here for your convenience
      (same as for part 1 of this assignment)

    while (!converged) {

        // compute score_new[vi] for all vertices belonging to this process
        score_new[vi] = sum over all vertices vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / totalVertices;

        score_new[vi] += sum over all nodes vj with no outgoing edges
                          { damping * score_old[vj] / totalVertices }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all vertices vi { abs(score_new[vi] - score_old[vi]) };
        converged = (global_diff < convergence)

        // Note that here, some communication between all the nodes is necessary
        // so that all nodes have the same copy of old scores before beginning the 
        // next iteration. You should be careful to make sure that any data you send 
        // is received before you delete or modify the buffers you are sending.

    }
    */
}
