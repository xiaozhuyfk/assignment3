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
    double out_scores[vertices_per_process]; 
    double all_vtx_scores[totalVertices];

    // initialize per-vertex scores
    #pragma omp parallel for
    for (int i = 0; i < vertices_per_process; ++i) {
        solution[i] = equal_prob;
    }

    bool converged = false;
    //initialize local disjoint set
    std::vector<int> disjoint;
    for (int i = 0 ; i < vertices_per_process ; i++) {
        if (!g.outgoing_edges[i].size()) {
            disjoint.push_back(i); //push global vertex index
        }
    }

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    double *old = (double *) malloc(sizeof(double) * vertices_per_process);
    int offset = g.world_rank * g.vertices_per_process;

    while (!converged) {
        std::memcpy(old, solution, sizeof(double) * vertices_per_process);
        std::vector<double*> disjoint_recv_bufs;

        MPI_Request* disjoint_send_reqs = new MPI_Request[g.world_size];

        // Phase 1 : update disjoint weight
        // Calculate local disjoint weight
        double disjoint_weight = 0.;
        #pragma omp parallel for reduction(+:disjoint_weight)
        for (std::size_t j = 0; j < disjoint.size(); j++) {
            disjoint_weight += damping * old[disjoint[j]] / totalVertices;
        }
        //pass local disjoint
        double* disjoint_send_buf = new double[1];
        double* disjoint_recv_buf = new double[1];

        if (g.world_rank) {
            disjoint_send_buf[0] = disjoint_weight;
            MPI_Isend(disjoint_send_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &disjoint_send_reqs[0]);
            MPI_Status status;
            MPI_Recv(disjoint_recv_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        } else {
            for (int i = 0; i < g.world_size; i++) {
                if (i!=g.world_rank) {
                    MPI_Status status;
                    MPI_Recv(disjoint_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                    disjoint_weight += disjoint_recv_buf[0];
                }
            }
            disjoint_send_buf[0] = disjoint_weight;
            for (int i = 1; i < g.world_size; i++) { //exclude self
                MPI_Isend(disjoint_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &disjoint_send_reqs[i]);
            }
        }
        // clear disjoint buf
        delete(disjoint_send_buf);
        delete(disjoint_recv_buf);
        delete(disjoint_send_reqs);

        // Phase 2 : send scores across machine
        std::vector<double*> recv_bufs;
        MPI_Request* send_reqs = new MPI_Request[g.world_size];

        std::map<Vertex, std::vector<double>> buf_map; // buffer to send to other worlds
        std::map<Vertex, double> score_map; // score to update to solution eventually

        //#pragma omp parallel for
        for (int i = 0; i < vertices_per_process; i++) {
            out_scores[i] = old[i] / static_cast<int>(g.outgoing_edges[i].size());
            //std::cout << out_scores[i] << std::endl;
        }

        /*
        for (int i = 0; i < vertices_per_process; i++) {
            double value = out_scores[i];
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
        }*/

        //#pragma omp parallel for
        for (int v = 0; v < g.incoming_edges.size();  v++) {
            //int rank = g.get_vertex_owner_rank(v);
            double sum = 0;
            for (int in = 0; in < g.incoming_edges[v].size(); in++) {
                sum += out_scores[g.incoming_edges[v][in]];
                //std::cout << " v is " << v << " " << sum << std::endl;
            }
            all_vtx_scores[v] = sum;
            //std::cout << all_vtx_scores[v] << std::endl;
        }
        //iterate through ranks to compile send
        //#pragma omp parallel for
        for (int r = 0 ; r < g.world_size ; r++) {
            std::vector<double> send_buf; 
            int out_offset = r * vertices_per_process;
            if (r==g.world_rank) {
                for (auto v : g.world_outgoing_map[r]) {
                    score_map[v-offset] = all_vtx_scores[v];
                    //std::cout << v-offset << " " <<  score_map[v-offset] << std::endl;
                }
            } else {
                for (auto v : g.world_outgoing_map[r]) {
                    send_buf.push_back((v- out_offset) * 1.0);
                    send_buf.push_back(all_vtx_scores[v]);
                }
                MPI_Isend(&send_buf,
                        static_cast<int> (send_buf.size()),
                        MPI_DOUBLE,
                        r, 0, MPI_COMM_WORLD, &send_reqs[r]);
            }
        }

        /*for (int v = 0; v < g.incoming_edges.size(); v++) {
            int rank = g.get_vertex_owner_rank(v);
            if (rank == g.world_rank) {
                score_map[v-offset] = sum;
            } else {
                score_map[]
            }
        }*/

        // initialize buffer size
        // some tips for casting vector to array
        /*
        #pragma omp parallel for
        for (int i = 0; i < g.world_size; i++) {
            if (i != g.world_rank && buf_map[i].size()) {
                double* send_buf = &buf_map[i][0];
                MPI_Isend(send_buf,
                    static_cast<int> (buf_map[i].size()),
                    MPI_DOUBLE,
                    i, 0, MPI_COMM_WORLD, &send_reqs[i]);
            }
        }*/

        // Receive and update value
        for (int i = 0; i < g.world_size; i++) {
            if (i!=g.world_rank && g.world_incoming_map[i].size()) {
                MPI_Status status;
                double* recv_buf = new double[g.world_incoming_map[i].size() * 2];
                recv_bufs.push_back(recv_buf);

                MPI_Recv(recv_buf, g.world_incoming_map[i].size() * 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status); //MPI_SOURCE?

                for(int j = 0; j < g.world_incoming_map[i].size(); j++) {
                    double value = recv_buf[2 * j + 1];
                    int recv_vertex = (int) recv_buf[2 * j];
                    score_map[recv_vertex] += value;
                }
            }
        }

        // Update the final value to solution to prepare for next iteration
        //#pragma omp parallel for
        for (int i = 0; i < vertices_per_process ; i++) {
            solution[i] = (damping * score_map[i]) + (1.0 - damping) /  totalVertices + disjoint_weight;
            //std::cout << "From : " << g.world_rank << " Vertex : " << i << " Score : " << solution[i] << std::endl;
        }

        //clear buf
        #pragma omp parallel for
        for (size_t i = 0; i < recv_bufs.size(); i++) {
            delete(recv_bufs[i]);
        }

        delete(send_reqs);

        // Phase 3 : Check for convergence
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
        double* converge_recv_buf = new double[1];


        if (g.world_rank){
            converge_send_buf[0] = diff;
            MPI_Isend(converge_send_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &converge_send_reqs[0]);
            MPI_Status status;
            MPI_Recv(converge_recv_buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            diff += converge_recv_buf[0];
        } else {
            for (int i = 1; i < g.world_size; i++) {
                MPI_Status status;
                MPI_Recv(converge_recv_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                diff += converge_recv_buf[0];
            }
            converge_send_buf[0] = diff;
            for (int i = 1; i < g.world_size; i++) { //exclude self
                MPI_Isend(converge_send_buf, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &converge_send_reqs[i]);
            }
        }
        converged = (diff < convergence);

        //clear converge buf
        delete(converge_send_buf);
        delete(converge_recv_buf);
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
