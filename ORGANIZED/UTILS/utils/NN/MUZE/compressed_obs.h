#ifndef COMPRESSED_OBS_H
#define COMPRESSED_OBS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compressed observation structure
typedef struct {
    // AABB/OBB bounding box information
    float center_x, center_y, center_z;
    float extent_x, extent_y, extent_z;
    float yaw, pitch, roll;  // Optional rotation
    
    // Silhouette tokens (ray hits/depth samples)
    int num_rays;
    float ray_distances[128];  // Up to 128 ray samples
    float ray_angles[128];     // Angles for each ray
    
    // Coarse voxel occupancy (16x16x8 around agent)
    int voxel_grid_size;
    uint8_t voxel_occupancy[2048]; // 16*16*8 = 2048
    
    // Agent state
    float health;
    float energy;
    int inventory_size;
    uint32_t game_time;
    
    // Tokenized local grid (8x8)
    int local_grid_size;
    uint8_t local_grid[64];  // 8x8 = 64 tokens
    
    // Observation dimension
    int obs_dim;
    
    // Total compressed size
    size_t total_size;
} CompressedObs;

// Compression functions
void compress_observation(float *raw_obs, int raw_size, CompressedObs *compressed);
void decompress_observation(CompressedObs *compressed, float *raw_obs, int raw_size);

// Vision encoder functions
void encode_aabb(float *vertices, int num_vertices, CompressedObs *compressed);
void encode_silhouette(float *depth_map, int width, int height, CompressedObs *compressed);
void encode_voxel_grid(float *voxels, int dim_x, int dim_y, int dim_z, CompressedObs *compressed);
void encode_local_grid(float *local_obs, int grid_size, CompressedObs *compressed);

// Utility functions
size_t get_compressed_size(void);
void init_compressed_obs(CompressedObs *compressed);
void cleanup_compressed_obs(CompressedObs *compressed);

#ifdef __cplusplus
}
#endif

#endif // COMPRESSED_OBS_H
