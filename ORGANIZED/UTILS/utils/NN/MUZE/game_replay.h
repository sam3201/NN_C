#ifndef GAME_REPLAY_H
#define GAME_REPLAY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int max_games;
  int max_steps;
  int obs_dim;
  int action_count;

  int game_count;
  int next_game;
  int cur_game;
  int cur_step;
  int in_episode;

  int *lengths;       /* [max_games] */
  float *obs_buf;     /* [max_games * max_steps * obs_dim] */
  float *pi_buf;      /* [max_games * max_steps * action_count] */
  int *a_buf;         /* [max_games * max_steps] */
  float *r_buf;       /* [max_games * max_steps] */
  int *done_buf;      /* [max_games * max_steps] */
  size_t *rb_idx_buf; /* [max_games * max_steps] */
} GameReplay;

GameReplay *gr_create(int max_games, int max_steps, int obs_dim,
                      int action_count);
void gr_free(GameReplay *gr);

void gr_start_episode(GameReplay *gr);
void gr_add_step(GameReplay *gr, const float *obs, const float *pi, int action,
                 float reward, int done, size_t rb_idx);
void gr_end_episode(GameReplay *gr);

int gr_game_count(const GameReplay *gr);
int gr_game_length(const GameReplay *gr, int game_idx);

int gr_save(const GameReplay *gr, const char *filename);
GameReplay *gr_load(const char *filename);

#ifdef __cplusplus
}
#endif

#endif
