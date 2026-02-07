// sam_muze_toy_test.c

#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"

#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // usleep

static void draw_env(int row, const ToyEnvState *env, int goal_pos) {
  for (int i = 0; i < (int)env->size; i++) {
    chtype ch = '.';
    if (i == goal_pos)
      ch = 'G';
    if (i == env->pos)
      ch = 'X';
    mvaddch(row, i, ch);
  }
}

static void draw_obs(int row, int col, const float *obs, int n) {
  mvprintw(row, col, "obs=[");
  for (int i = 0; i < n; i++) {
    printw("%.0f", obs[i]);
    if (i + 1 < n)
      printw(" ");
  }
  printw("]");
}

static void cleanup_and_exit(SAM_t *sam, MuCortex *cortex, int exit_code) {
  if (cortex) {
    if (cortex->mcts_model) {
      mu_model_free_toy(cortex->mcts_model);
      cortex->mcts_model = NULL;
    }
    SAM_MUZE_destroy(cortex);
  }
  if (sam)
    SAM_destroy(sam);
  endwin();
  exit(exit_code);
}

int main(void) {
  srand(1);

  // --- init curses ---
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  keypad(stdscr, TRUE);
  nodelay(stdscr, TRUE);

  ToyEnvState env = {.pos = 0, .size = 8};
  const size_t obs_dim = env.size;
  const int action_count = 2;
  const int goal_pos = (int)env.size - 1;

  SAM_t *sam = SAM_init(obs_dim, (size_t)action_count, 2, 0);
  if (!sam) {
    endwin();
    fprintf(stderr, "SAM_init failed\n");
    return 1;
  }

  MuCortex *cortex = SAM_as_MUZE(sam);
  if (!cortex) {
    SAM_destroy(sam);
    endwin();
    fprintf(stderr, "SAM_as_MUZE failed\n");
    return 1;
  }

  // --- Enable MCTS with toy MuZero model ---
  cortex->use_mcts = true;
  cortex->mcts_model = mu_model_create_toy((int)obs_dim, action_count);
  if (!cortex->mcts_model) {
    fprintf(stderr, "mu_model_create_toy failed\n");
    cleanup_and_exit(sam, cortex, 1);
  }

  // Deterministic-ish parameters for the toy line world
  cortex->mcts_params.num_simulations = 64;
  cortex->mcts_params.max_depth = 16;
  cortex->mcts_params.c_puct = 1.25f;
  cortex->mcts_params.discount = 1.0f;
  cortex->mcts_params.temperature = 0.0f; // greedy from visits
  cortex->mcts_params.dirichlet_alpha = 0.0f;
  cortex->mcts_params.dirichlet_eps = 0.0f;

  const int episodes = 50;
  const int max_steps = 128;

  // Use obs_dim for stack arrays (C99 VLA); avoids hardcoding 8.
  // If your compiler is in C11 mode with VLAs disabled, compile with -std=c99,
  // or swap these to malloc/free.
  for (int ep = 0; ep < episodes; ep++) {
    float obs[obs_dim];
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;

    for (int step = 0; step < max_steps; step++) {
      int key = getch();
      if (key == 'q' || key == 'Q')
        cleanup_and_exit(sam, cortex, 0);

      int action = muze_plan(cortex, obs, obs_dim, (size_t)action_count);

      float next_obs[obs_dim];
      float env_reward = 0.0f;
      int env_done = 0;

      if (toy_env_step(&env, action, next_obs, &env_reward, &env_done) != 0) {
        mvprintw(10, 0, "toy_env_step error");
        refresh();
        usleep(300 * 1000);
        break;
      }

      // Learn from env's reward/done (toy model's dynamics also matches this)
      if (cortex->learn)
        cortex->learn(cortex->brain, obs, obs_dim, action, env_reward,
                      env_done);

      ep_return += env_reward;
      memcpy(obs, next_obs, sizeof(next_obs));

      clear();
      mvprintw(1, 0, "MUZE toy + MCTS (q to quit)");
      mvprintw(2, 0, "ep=%d/%d  step=%d/%d  action=%d  reward=%.1f  done=%d",
               ep, episodes - 1, step, max_steps - 1, action, env_reward,
               env_done);
      mvprintw(3, 0, "pos=%d  goal=%d  return=%.1f", env.pos, goal_pos,
               ep_return);

      draw_env(5, &env, goal_pos);
      draw_obs(7, 0, obs, (int)obs_dim);

      refresh();
      usleep(60 * 1000);

      if (env_done)
        break;
    }

    mvprintw(9, 0, "episode %d return=%.1f final_pos=%d  (press any key)", ep,
             ep_return, env.pos);
    nodelay(stdscr, FALSE);
    getch();
    nodelay(stdscr, TRUE);
  }

  cleanup_and_exit(sam, cortex, 0);
  return 0;
}
