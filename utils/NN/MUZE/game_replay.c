#include "game_replay.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  GR_MAGIC = 0x47525631u, /* "GRV1" */
  GR_VERSION = 1u
};

static size_t gr_offset(const GameReplay *gr, int game_idx, int step) {
  return ((size_t)game_idx * (size_t)gr->max_steps + (size_t)step);
}

GameReplay *gr_create(int max_games, int max_steps, int obs_dim,
                      int action_count) {
  if (max_games <= 0 || max_steps <= 0 || obs_dim <= 0 || action_count <= 0)
    return NULL;

  GameReplay *gr = (GameReplay *)calloc(1, sizeof(GameReplay));
  if (!gr)
    return NULL;

  gr->max_games = max_games;
  gr->max_steps = max_steps;
  gr->obs_dim = obs_dim;
  gr->action_count = action_count;
  gr->game_count = 0;
  gr->next_game = 0;
  gr->cur_game = -1;
  gr->cur_step = 0;
  gr->in_episode = 0;

  gr->lengths = (int *)calloc((size_t)max_games, sizeof(int));
  gr->obs_buf = (float *)malloc(sizeof(float) * (size_t)max_games *
                                (size_t)max_steps * (size_t)obs_dim);
  gr->pi_buf = (float *)malloc(sizeof(float) * (size_t)max_games *
                               (size_t)max_steps * (size_t)action_count);
  gr->a_buf =
      (int *)malloc(sizeof(int) * (size_t)max_games * (size_t)max_steps);
  gr->r_buf =
      (float *)malloc(sizeof(float) * (size_t)max_games * (size_t)max_steps);
  gr->done_buf =
      (int *)malloc(sizeof(int) * (size_t)max_games * (size_t)max_steps);
  gr->rb_idx_buf = (size_t *)malloc(sizeof(size_t) * (size_t)max_games *
                                    (size_t)max_steps);

  if (!gr->lengths || !gr->obs_buf || !gr->pi_buf || !gr->a_buf ||
      !gr->r_buf || !gr->done_buf || !gr->rb_idx_buf) {
    gr_free(gr);
    return NULL;
  }

  memset(gr->obs_buf, 0, sizeof(float) * (size_t)max_games *
                             (size_t)max_steps * (size_t)obs_dim);
  memset(gr->pi_buf, 0, sizeof(float) * (size_t)max_games *
                            (size_t)max_steps * (size_t)action_count);
  memset(gr->a_buf, 0,
         sizeof(int) * (size_t)max_games * (size_t)max_steps);
  memset(gr->r_buf, 0,
         sizeof(float) * (size_t)max_games * (size_t)max_steps);
  memset(gr->done_buf, 0,
         sizeof(int) * (size_t)max_games * (size_t)max_steps);
  memset(gr->rb_idx_buf, 0,
         sizeof(size_t) * (size_t)max_games * (size_t)max_steps);

  return gr;
}

void gr_free(GameReplay *gr) {
  if (!gr)
    return;
  free(gr->lengths);
  free(gr->obs_buf);
  free(gr->pi_buf);
  free(gr->a_buf);
  free(gr->r_buf);
  free(gr->done_buf);
  free(gr->rb_idx_buf);
  free(gr);
}

void gr_start_episode(GameReplay *gr) {
  if (!gr)
    return;
  gr->cur_game = gr->next_game;
  gr->cur_step = 0;
  gr->in_episode = 1;
  gr->lengths[gr->cur_game] = 0;
}

void gr_add_step(GameReplay *gr, const float *obs, const float *pi, int action,
                 float reward, int done, size_t rb_idx) {
  if (!gr || !gr->in_episode || !obs || !pi)
    return;
  if (gr->cur_game < 0 || gr->cur_step >= gr->max_steps)
    return;

  size_t off = gr_offset(gr, gr->cur_game, gr->cur_step);
  memcpy(gr->obs_buf + off * (size_t)gr->obs_dim, obs,
         sizeof(float) * (size_t)gr->obs_dim);
  memcpy(gr->pi_buf + off * (size_t)gr->action_count, pi,
         sizeof(float) * (size_t)gr->action_count);
  gr->a_buf[off] = action;
  gr->r_buf[off] = reward;
  gr->done_buf[off] = done ? 1 : 0;
  gr->rb_idx_buf[off] = rb_idx;

  gr->cur_step++;
  gr->lengths[gr->cur_game] = gr->cur_step;

  if (done || gr->cur_step >= gr->max_steps)
    gr_end_episode(gr);
}

void gr_end_episode(GameReplay *gr) {
  if (!gr || !gr->in_episode)
    return;
  gr->in_episode = 0;
  if (gr->game_count < gr->max_games)
    gr->game_count++;
  gr->next_game = (gr->next_game + 1) % gr->max_games;
  gr->cur_game = -1;
  gr->cur_step = 0;
}

int gr_game_count(const GameReplay *gr) {
  if (!gr)
    return 0;
  return gr->game_count;
}

int gr_game_length(const GameReplay *gr, int game_idx) {
  if (!gr || game_idx < 0 || game_idx >= gr->max_games)
    return 0;
  return gr->lengths[game_idx];
}

int gr_save(const GameReplay *gr, const char *filename) {
  if (!gr || !filename)
    return 0;

  FILE *f = fopen(filename, "wb");
  if (!f)
    return 0;

  uint32_t magic = GR_MAGIC;
  uint32_t version = GR_VERSION;
  if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
      fwrite(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (fwrite(&gr->max_games, sizeof(int), 1, f) != 1 ||
      fwrite(&gr->max_steps, sizeof(int), 1, f) != 1 ||
      fwrite(&gr->obs_dim, sizeof(int), 1, f) != 1 ||
      fwrite(&gr->action_count, sizeof(int), 1, f) != 1 ||
      fwrite(&gr->game_count, sizeof(int), 1, f) != 1 ||
      fwrite(&gr->next_game, sizeof(int), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  size_t games = (size_t)gr->max_games;
  size_t steps = (size_t)gr->max_steps;
  size_t obs_bytes = sizeof(float) * games * steps * (size_t)gr->obs_dim;
  size_t pi_bytes = sizeof(float) * games * steps * (size_t)gr->action_count;
  size_t a_bytes = sizeof(int) * games * steps;
  size_t r_bytes = sizeof(float) * games * steps;
  size_t done_bytes = sizeof(int) * games * steps;
  size_t idx_bytes = sizeof(size_t) * games * steps;

  if (fwrite(gr->lengths, sizeof(int), games, f) != games ||
      fwrite(gr->obs_buf, 1, obs_bytes, f) != obs_bytes ||
      fwrite(gr->pi_buf, 1, pi_bytes, f) != pi_bytes ||
      fwrite(gr->a_buf, 1, a_bytes, f) != a_bytes ||
      fwrite(gr->r_buf, 1, r_bytes, f) != r_bytes ||
      fwrite(gr->done_buf, 1, done_bytes, f) != done_bytes ||
      fwrite(gr->rb_idx_buf, 1, idx_bytes, f) != idx_bytes) {
    fclose(f);
    return 0;
  }

  fclose(f);
  return 1;
}

GameReplay *gr_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;

  uint32_t magic = 0;
  uint32_t version = 0;
  if (fread(&magic, sizeof(magic), 1, f) != 1 ||
      fread(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  if (magic != GR_MAGIC || version != GR_VERSION) {
    fclose(f);
    return NULL;
  }

  int max_games = 0;
  int max_steps = 0;
  int obs_dim = 0;
  int action_count = 0;
  int game_count = 0;
  int next_game = 0;

  if (fread(&max_games, sizeof(int), 1, f) != 1 ||
      fread(&max_steps, sizeof(int), 1, f) != 1 ||
      fread(&obs_dim, sizeof(int), 1, f) != 1 ||
      fread(&action_count, sizeof(int), 1, f) != 1 ||
      fread(&game_count, sizeof(int), 1, f) != 1 ||
      fread(&next_game, sizeof(int), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  GameReplay *gr = gr_create(max_games, max_steps, obs_dim, action_count);
  if (!gr) {
    fclose(f);
    return NULL;
  }

  gr->game_count = game_count;
  gr->next_game = next_game;
  gr->cur_game = -1;
  gr->cur_step = 0;
  gr->in_episode = 0;

  size_t games = (size_t)max_games;
  size_t steps = (size_t)max_steps;
  size_t obs_bytes = sizeof(float) * games * steps * (size_t)obs_dim;
  size_t pi_bytes = sizeof(float) * games * steps * (size_t)action_count;
  size_t a_bytes = sizeof(int) * games * steps;
  size_t r_bytes = sizeof(float) * games * steps;
  size_t done_bytes = sizeof(int) * games * steps;
  size_t idx_bytes = sizeof(size_t) * games * steps;

  if (fread(gr->lengths, sizeof(int), games, f) != games ||
      fread(gr->obs_buf, 1, obs_bytes, f) != obs_bytes ||
      fread(gr->pi_buf, 1, pi_bytes, f) != pi_bytes ||
      fread(gr->a_buf, 1, a_bytes, f) != a_bytes ||
      fread(gr->r_buf, 1, r_bytes, f) != r_bytes ||
      fread(gr->done_buf, 1, done_bytes, f) != done_bytes ||
      fread(gr->rb_idx_buf, 1, idx_bytes, f) != idx_bytes) {
    gr_free(gr);
    fclose(f);
    return NULL;
  }

  fclose(f);
  return gr;
}
