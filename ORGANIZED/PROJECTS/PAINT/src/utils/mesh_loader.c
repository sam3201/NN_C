#include "mesh_loader.h"
#include <string.h>

int load_custom_mesh(const char *filename, MeshData *mesh) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open mesh file: %s\n", filename);
        return 0;
    }
    
    char header[4];
    fread(header, 4, 1, file);
    if (strncmp(header, "MESH", 4) != 0) {
        printf("Invalid mesh file format\n");
        fclose(file);
        return 0;
    }
    
    fread(&mesh->vertex_count, sizeof(int), 1, file);
    fread(&mesh->index_count, sizeof(int), 1, file);
    
    mesh->vertices = malloc(sizeof(Vertex) * mesh->vertex_count);
    mesh->indices = malloc(sizeof(unsigned int) * mesh->index_count);
    
    if (!mesh->vertices || !mesh->indices) {
        printf("Failed to allocate memory for mesh\n");
        if (mesh->vertices) free(mesh->vertices);
        if (mesh->indices) free(mesh->indices);
        fclose(file);
        return 0;
    }
    
    fread(mesh->vertices, sizeof(Vertex), mesh->vertex_count, file);
    fread(mesh->indices, sizeof(unsigned int), mesh->index_count, file);
    
    fclose(file);
    printf("Successfully loaded mesh: %d vertices, %d indices\n", 
           mesh->vertex_count, mesh->index_count);
    return 1;
}

void free_mesh_data(MeshData *mesh) {
    if (mesh->vertices) free(mesh->vertices);
    if (mesh->indices) free(mesh->indices);
    mesh->vertices = NULL;
    mesh->indices = NULL;
    mesh->vertex_count = 0;
    mesh->index_count = 0;
}

// Simple function to convert our custom mesh to the game's mesh format
int convert_to_game_mesh(const MeshData *custom_mesh, float **out_verts, int *vcount, 
                        unsigned int **out_indices, int *icount) {
    *vcount = custom_mesh->vertex_count;
    *icount = custom_mesh->index_count;
    
    *out_verts = malloc(sizeof(float) * custom_mesh->vertex_count * 6);
    *out_indices = malloc(sizeof(unsigned int) * custom_mesh->index_count);
    
    if (!*out_verts || !*out_indices) {
        if (*out_verts) free(*out_verts);
        if (*out_indices) free(*out_indices);
        return 0;
    }
    
    // Convert vertices to game format (pos(3) + normal(3))
    for (int i = 0; i < custom_mesh->vertex_count; i++) {
        (*out_verts)[i * 6 + 0] = custom_mesh->vertices[i].x;
        (*out_verts)[i * 6 + 1] = custom_mesh->vertices[i].y;
        (*out_verts)[i * 6 + 2] = custom_mesh->vertices[i].z;
        (*out_verts)[i * 6 + 3] = custom_mesh->vertices[i].nx;
        (*out_verts)[i * 6 + 4] = custom_mesh->vertices[i].ny;
        (*out_verts)[i * 6 + 5] = custom_mesh->vertices[i].nz;
    }
    
    memcpy(*out_indices, custom_mesh->indices, 
           sizeof(unsigned int) * custom_mesh->index_count);
    
    return 1;
}
