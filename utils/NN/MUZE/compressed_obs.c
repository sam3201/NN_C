#include "compressed_obs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Initialize compressed observation structure
void init_compressed_obs(CompressedObs *compressed) {
    if (!compressed) return;
    
    memset(compressed, 0, sizeof(CompressedObs));
    compressed->num_rays = 128;
    compressed->voxel_grid_size = 16;
    compressed->local_grid_size = 8;
    compressed->obs_dim = sizeof(CompressedObs) / sizeof(float);
}

// Clean up compressed observation
void cleanup_compressed_obs(CompressedObs *compressed) {
    if (!compressed) return;
    // No dynamic allocation in current structure
}

// Get compressed observation size
size_t get_compressed_size(void) {
    return sizeof(CompressedObs);
}

// Compute AABB from vertices
void compute_aabb(float *vertices, int num_vertices, float *center_x, float *center_y, 
                  float *center_z, float *extent_x, float *extent_y, float *extent_z) {
    if (!vertices || num_vertices == 0) return;
    
    float min_x = vertices[0], max_x = vertices[0];
    float min_y = vertices[1], max_y = vertices[1];
    float min_z = vertices[2], max_z = vertices[2];
    
    for (int i = 1; i < num_vertices; i++) {
        float x = vertices[i * 3];
        float y = vertices[i * 3 + 1];
        float z = vertices[i * 3 + 2];
        
        min_x = fminf(min_x, x);
        max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y);
        max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z);
        max_z = fmaxf(max_z, z);
    }
    
    *center_x = (min_x + max_x) / 2.0f;
    *center_y = (min_y + max_y) / 2.0f;
    *center_z = (min_z + max_z) / 2.0f;
    *extent_x = (max_x - min_x) / 2.0f;
    *extent_y = (max_y - min_y) / 2.0f;
    *extent_z = (max_z - min_z) / 2.0f;
}

// Encode AABB/OBB bounding box
void encode_aabb(float *vertices, int num_vertices, CompressedObs *compressed) {
    if (!vertices || !compressed) return;
    
    compute_aabb(vertices, num_vertices, 
                  &compressed->center_x, &compressed->center_y, &compressed->center_z,
                  &compressed->extent_x, &compressed->extent_y, &compressed->extent_z);
}

// Encode silhouette using ray casting
void encode_silhouette(float *depth_map, int width, int height, CompressedObs *compressed) {
    if (!depth_map || !compressed) return;
    
    // Cast rays from center in uniform pattern
    int center_x = width / 2;
    int center_y = height / 2;
    int max_radius = fminf(width, height) / 2;
    
    compressed->num_rays = 128;
    
    for (int i = 0; i < 128; i++) {
        float angle = 2.0f * M_PI * i / 128.0f;
        float dx = cosf(angle) * max_radius;
        float dy = sinf(angle) * max_radius;
        
        int x = center_x + (int)dx;
        int y = center_y + (int)dy;
        
        // Clamp to image bounds
        x = fmaxf(0, fminf(width - 1, x));
        y = fmaxf(0, fminf(height - 1, y));
        
        compressed->ray_distances[i] = depth_map[y * width + x];
        compressed->ray_angles[i] = angle;
    }
}

// Encode coarse voxel grid around agent
void encode_voxel_grid(float *voxels, int dim_x, int dim_y, int dim_z, CompressedObs *compressed) {
    if (!voxels || !compressed) return;
    
    compressed->voxel_grid_size = 16;
    compressed->total_size = sizeof(CompressedObs);
    
    // Sample coarse grid (16x16x8) from full voxel grid
    int agent_x = dim_x / 2;
    int agent_y = dim_y / 2;
    int agent_z = dim_z / 2;
    
    int half_size_x = 8;
    int half_size_y = 8;
    int half_size_z = 4;
    
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 8; z++) {
                int voxel_x = agent_x - half_size_x + x;
                int voxel_y = agent_y - half_size_y + y;
                int voxel_z = agent_z - half_size_z + z;
                
                // Check bounds
                if (voxel_x >= 0 && voxel_x < dim_x && 
                    voxel_y >= 0 && voxel_y < dim_y && 
                    voxel_z >= 0 && voxel_z < dim_z) {
                    
                    // Simple occupancy: 1 if occupied, 0 if empty
                    float voxel_value = voxels[voxel_z * dim_x * dim_y + voxel_y * dim_x + voxel_x];
                    compressed->voxel_occupancy[z * 16 * 16 + y * 16 + x] = (voxel_value > 0.5f) ? 1 : 0;
                } else {
                    compressed->voxel_occupancy[z * 16 * 16 + y * 16 + x] = 0;
                }
            }
        }
    }
}

// Tokenize local observation grid
void encode_local_grid(float *local_obs, int grid_size, CompressedObs *compressed) {
    if (!local_obs || !compressed) return;
    
    compressed->local_grid_size = grid_size;
    compressed->total_size = sizeof(CompressedObs);
    
    // Simple tokenization: quantize to 8 levels
    for (int i = 0; i < grid_size * grid_size; i++) {
        float value = local_obs[i];
        uint8_t token = (uint8_t)(fminf(7.0f, fmaxf(0.0f, value * 8.0f)));
        compressed->local_grid[i] = token;
    }
}

// Main compression function
void compress_observation(float *raw_obs, int raw_size, CompressedObs *compressed) {
    if (!raw_obs || !compressed) return;
    
    init_compressed_obs(compressed);
    
    // For now, assume raw_obs is already in compressed format
    // In practice, this would analyze the raw observation and extract features
    
    // Extract agent state (assuming it's part of raw_obs)
    if (raw_size >= 4) {
        compressed->health = raw_obs[0];
        compressed->energy = raw_obs[1];
        compressed->inventory_size = (int)raw_obs[2];
        compressed->game_time = (uint32_t)raw_obs[3];
    }
    
    // Store compressed size
    compressed->total_size = sizeof(CompressedObs);
}

// Decompress observation (reverse of compression)
void decompress_observation(CompressedObs *compressed, float *raw_obs, int raw_size) {
    if (!compressed || !raw_obs) return;
    
    // For now, just copy the compressed data back
    // In practice, this would reconstruct the full observation
    
    if (raw_size >= 4) {
        raw_obs[0] = compressed->health;
        raw_obs[1] = compressed->energy;
        raw_obs[2] = (float)compressed->inventory_size;
        raw_obs[3] = (float)compressed->game_time;
    }
    
    // Zero out the rest
    for (int i = 4; i < raw_size; i++) {
        raw_obs[i] = 0.0f;
    }
}
