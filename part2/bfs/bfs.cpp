#include <cstring>
#include <set>
#include <iostream>
#include <vector>
#include <queue>
#include "bfs.h"

#define contains(container, element) \
  (container.find(element) != container.end())

/**
 *
 * global_frontier_sync--
 *
 * Takes a distributed graph, and a distributed frontier with each node containing
 * world_size independently produced new frontiers, and merges them such that each
 * node holds the subset of the global frontier containing local vertices.
 */
void global_frontier_sync(DistGraph &g, DistFrontier &frontier, int *depths) {

    // TODO 15-418/618 STUDENTS
    //
    // In this function, you should synchronize between all nodes you
    // are using for your computation. This would mean sending and
    // receiving data between nodes in a manner you see fit. Note for
    // those using async sends: you should be careful to make sure that
    // any data you send is received before you delete or modify the
    // buffers you are sending.

    int world_size = g.world_size;
    int world_rank = g.world_rank;
    int offset = g.start_vertex;

    std::vector<int*> send_bufs;
    std::vector<int> send_idx;
    std::vector<int*> recv_bufs;
    std::set<int> recv_vertex_set;
    std::vector<int> recv_not_updated;

    MPI_Request* send_reqs = new MPI_Request[world_size];
    MPI_Status* probe_status = new MPI_Status[world_size];

    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            int* send_buf = new int[frontier.sizes[i] * 2];
            //std::cout << "send_buf size" << frontier.sizes[i] << "of array " << i <<std::endl;

            send_bufs.push_back(send_buf);
            send_idx.push_back(i);

            for (int j = 0; j < frontier.sizes[i]; j++) {
                send_buf[2*j] = frontier.elements[i][j];
                send_buf[2*j+1] = frontier.depths[i][j];
                //std::cout << "send " << send_buf[2*j] << std::endl;
            }

            MPI_Isend(send_buf, frontier.sizes[i] * 2, MPI_INT,
                      i, 0, MPI_COMM_WORLD, &send_reqs[i]);
        }
    }
    // Receive from all other buffers
    for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &probe_status[i]);
            int num_vals = 0;
            MPI_Get_count(&probe_status[i], MPI_INT, &num_vals);

            int* recv_buf = new int[num_vals];
            recv_bufs.push_back(recv_buf);
            MPI_Recv(recv_buf, num_vals, MPI_INT, probe_status[i].MPI_SOURCE,
                     probe_status[i].MPI_TAG, MPI_COMM_WORLD, &status);
            for (int j = 0; j < num_vals; j+=2) {
                assert(g.get_vertex_owner_rank(recv_buf[j]) == world_rank);
                Vertex v = recv_buf[j];
                if (depths[v-offset] == NOT_VISITED_MARKER && recv_vertex_set.find(v) == recv_vertex_set.end()) {
                    recv_vertex_set.insert(v);
                    recv_not_updated.push_back(v);
                    recv_not_updated.push_back(recv_buf[j+1]);
                    //std::cout << "receive " << recv_buf[j] << " with depth " << recv_buf[j+1] << std::endl;
                }
            }
        }
    }
    // Make sure all the Isends are received before local modification.
    for (size_t i = 0; i < send_bufs.size(); i++) {
        MPI_Status status;
        MPI_Wait(&send_reqs[send_idx[i]], &status);
        delete(send_bufs[i]);
    }
    for (size_t i = 0; i < recv_bufs.size(); i++) {
        delete(recv_bufs[i]);
    }
    delete(send_reqs);
    delete(probe_status);

    for (std::vector<int>::size_type j = 0; j != recv_not_updated.size(); j+=2) {
        frontier.add(world_rank, recv_not_updated[j], recv_not_updated[j+1]);
        depths[recv_not_updated[j]-offset] = recv_not_updated[j+1];
    }

}

/*
 * bfs_step --
 *
 * Carry out one step of a distributed bfs
 *
 * depths: current state of depths array for local vertices
 * current_frontier/next_frontier: copies of the distributed frontier structure
 *
 * NOTE TO STUDENTS: We gave you this function as a stub.  Feel free
 * to change as you please (including the arguments)
 */
void bfs_step(DistGraph &g, int *depths,
        DistFrontier &current_frontier,
        DistFrontier &next_frontier) {
    int offset = g.start_vertex;
    int frontier_size = current_frontier.get_local_frontier_size();
    Vertex* local_frontier = current_frontier.get_local_frontier();
    std::set<Vertex> query_frontier_set;

    // keep in mind, this node owns the vertices with global ids:
    // g.start_vertex, g.start_vertex+1, g.start_vertex+2, etc...

    // TODO 15-418/618 STUDENTS
    //
    // implement a step of the BFS

    /*
     * TODO: use top-down method
     */
    for (int i = 0; i < frontier_size; i++) {
        int local_idx = local_frontier[i]-offset;
        int new_depth = depths[local_idx]+1;
        for (auto& dest : g.outgoing_edges[local_idx]) {
            int rank = g.get_vertex_owner_rank(dest);
            //std::cout << rank << " " << dest << std::endl;
            if (rank == g.world_rank && depths[dest-offset] == NOT_VISITED_MARKER) {
                // A local vertex unvisited
                next_frontier.add(g.world_rank, dest, new_depth);
                depths[dest-offset] = new_depth;
                //std::cout << "From " << i+offset << " to " << dest << " with depth " << depths[dest-offset] << std::endl;
            } else if (rank != g.world_rank && query_frontier_set.find(dest) == query_frontier_set.end()) {
                // A potential new vertex in other machine
                query_frontier_set.insert(dest);
                next_frontier.add(g.get_vertex_owner_rank(dest), dest, new_depth);
                //std::cout << "From " << i+offset << " to " << dest << " with depth " << new_depth << std::endl;
                //std::cout << next_frontier.sizes[g.get_vertex_owner_rank(dest)] << std::endl;
            }
        }
    }
}

/*
 * bfs --
 *
 * Execute a distributed BFS on the distributed graph g
 *
 * Upon return, depths[i] should be the distance of the i'th local
 * vertex from the BFS root node
 */
void bfs(DistGraph &g, int *depths) {
    DistFrontier current_frontier(g.vertices_per_process, g.world_size,
            g.world_rank);
    DistFrontier next_frontier(g.vertices_per_process, g.world_size,
            g.world_rank);

    DistFrontier *cur_front = &current_frontier,
            *next_front = &next_frontier;

    // Initialize all the depths to NOT_VISITED_MARKER.
    // Note: Only storing local vertex depths.
    for (int i = 0; i < g.vertices_per_process; ++i)
        depths[i] = NOT_VISITED_MARKER;

    // Add the root node to the frontier
    int offset = g.start_vertex;
    if (g.get_vertex_owner_rank(ROOT_NODE_ID) == g.world_rank) {
        current_frontier.add(g.get_vertex_owner_rank(ROOT_NODE_ID), ROOT_NODE_ID, 0);
        depths[ROOT_NODE_ID - offset] = 0;
    }
    //int counter = 0;
    while (true) {
        
        //counter++;
        //if (!g.world_rank){
            //std::cout << counter << std::endl;
        //}

        bfs_step(g, depths, *cur_front, *next_front);

        // this is a global empty check, not a local frontier empty check.
        // You will need to implement is_empty() in ../dist_graph.h
        if (next_front->is_empty())
            break;

        // exchange frontier information
        global_frontier_sync(g, *next_front, depths);

        DistFrontier *temp = cur_front;
        cur_front = next_front;
        next_front = temp;
        next_front->clear();
    }
}

