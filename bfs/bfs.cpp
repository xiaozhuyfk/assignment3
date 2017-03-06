#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <set>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*) malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
        Graph g,
        vertex_set* frontier,
        vertex_set* new_frontier,
        int* distances) {

    int num_threads = omp_get_max_threads();
    int **dist_frontier = (int **) malloc(sizeof(int *) * num_threads);
    int frontier_size[num_threads];
    memset(frontier_size, 0, num_threads * sizeof(int));

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        dist_frontier[i] = (int *) malloc(sizeof(int) * g->num_nodes);

        #pragma omp for
        for (int idx = 0; idx < frontier->count; idx++) {
            int node = frontier->vertices[idx];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ?
                    g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] != NOT_VISITED_MARKER) continue;

                if (__sync_bool_compare_and_swap(
                        &distances[outgoing],
                        NOT_VISITED_MARKER,
                        distances[node] + 1)) {
                    dist_frontier[i][frontier_size[i]++] = outgoing;
                }
            }
        }

        #pragma omp critical
        {
            int count = frontier_size[i];
            memcpy(new_frontier->vertices + new_frontier->count,
                    dist_frontier[i],
                    sizeof(int) * count);
            new_frontier->count += count;
        }
    }

    /*
    for (int i = 0; i < num_threads; i++) {
        int count = frontier_size[i];
        memcpy(new_frontier->vertices + new_frontier->count,
                dist_frontier[i],
                sizeof(int) * count);
        new_frontier->count += count;
    }
    */

    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        free(dist_frontier[i]);
    }
    free(dist_frontier);
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

int bottom_up_step(
        Graph g,
        int distance,
        int* distances) {

    int num_threads = omp_get_max_threads();
    int frontier_size[num_threads];
    memset(frontier_size, 0, num_threads * sizeof(int));

    #pragma omp parallel
    {
        int i = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int node = 0; node < g->num_nodes; node++) {
            if (distances[node] == NOT_VISITED_MARKER) {
                const Vertex* start = incoming_begin(g, node);
                const Vertex* end = incoming_end(g, node);
                for (const Vertex *v = start; v != end; v++) {
                    Vertex in = *v;
                    if (distances[in] == distance) {
                        distances[node] = distances[in] + 1;
                        frontier_size[i]++;
                        break;
                    }
                }
            }
        }
    }

    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_threads; i++) {
        int count = frontier_size[i];
        sum += count;
    }

    return sum;
}

void hybrid_bottom_up_step(
        Graph g,
        int distance,
        vertex_set *new_frontier,
        int* distances) {

    int num_threads = omp_get_max_threads();
    int **dist_frontier = (int **) malloc(sizeof(int *) * num_threads);
    int frontier_size[num_threads];
    memset(frontier_size, 0, num_threads * sizeof(int));

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        dist_frontier[i] = (int *) malloc(sizeof(int) * g->num_nodes);

        #pragma omp for schedule(static)
        for (int node = 0; node < g->num_nodes; node++) {
            if (distances[node] == NOT_VISITED_MARKER) {
                const Vertex* start = incoming_begin(g, node);
                const Vertex* end = incoming_end(g, node);
                for (const Vertex *v = start; v != end; v++) {
                    Vertex in = *v;
                    if (distances[in] != distance) continue;

                    if (distances[in] == distance) {
                        if (__sync_bool_compare_and_swap(
                                &distances[node],
                                NOT_VISITED_MARKER,
                                distances[in] + 1)) {
                            dist_frontier[i][frontier_size[i]++] = node;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < num_threads; i++) {
        int count = frontier_size[i];
        memcpy(new_frontier->vertices + new_frontier->count,
                dist_frontier[i],
                sizeof(int) * count);
        new_frontier->count += count;
    }

    #pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        free(dist_frontier[i]);
    }
    free(dist_frontier);
}

void bfs_bottom_up(Graph graph, solution* sol) {
    // 15-418/618 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int distance = 0;

    while (true) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        if (bottom_up_step(graph, distance, sol->distances) == 0) break;
        distance++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution* sol) {
    // 15-418/618 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int distance = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        if (frontier->count < 1000000) {
            top_down_step(graph, frontier, new_frontier, sol->distances);
        } else {
            hybrid_bottom_up_step(graph, distance, new_frontier, sol->distances);
        }
        distance++;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
