#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x, y, z;
    float nx, ny, nz;
} Vertex;

typedef struct {
    Vertex *vertices;
    unsigned int *indices;
    int vertex_count;
    int index_count;
} MeshData;

// Function declarations
int load_custom_mesh(const char *filename, MeshData *mesh);
void free_mesh_data(MeshData *mesh);
int convert_to_game_mesh(const MeshData *custom_mesh, float **out_verts, int *vcount, 
                        unsigned int **out_indices, int *icount);

#endif // MESH_LOADER_H
