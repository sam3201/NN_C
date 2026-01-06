#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"

#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // usleep

static void draw_env(int row, const ToyEnvState *env, int goal_pos) {
  // Track
  for (int i = 0; i < (int)env->size; i++) {
    chtype ch = ' ';
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
    printf("SAM_init failed\n");
    return 1;
  }

  MuCortex *cortex = SAM_as_MUZE(sam);
  if (!cortex) {
    SAM_destroy(sam);
    endwin();
    printf("SAM_as_MUZE failed\n");
    return 1;
  }

  const int episodes = 50;
  const int max_steps = 128;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[8]; // env.size == 8 here; if you change env.size, adjust
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;

    for (int step = 0; step < max_steps; step++) {
      // allow quitting
      int key = getch();
      if (key == 'q' || key == 'Q') {
        SAM_MUZE_destroy(cortex);
        SAM_destroy(sam);
        endwin();
        return 0;
      }

      int action = muze_plan(cortex, obs, obs_dim, (size_t)action_count);

      float next_obs[8];
      float env_reward = 0.0f;
      int env_done = 0;

      if (toy_env_step(&env, action, next_obs, &env_reward, &env_done) != 0) {
        break;
      }

      // learn uses env-provided reward/done
      cortex->learn(cortex->brain, obs, obs_dim, action, env_reward, env_done);

      ep_return += env_reward;
      memcpy(obs, next_obs, sizeof(next_obs));

      // draw
      clear();
      mvprintw(1, 0, "ep=%d/%d  step=%d/%d  action=%d  reward=%.1f  done=%d",
               ep, episodes - 1, step, max_steps - 1, action, env_reward,
               env_done);
      mvprintw(2, 0, "pos=%d  goal=%d  return=%.1f", env.pos, goal_pos,
               ep_return);

      draw_env(4, &env, goal_pos);
      draw_obs(6, 0, obs, (int)obs_dim);

      refresh();
      usleep(60 * 1000); // 60ms

      if (env_done)
        break;
    }

    // episode summary pause
    mvprintw(8, 0, "episode %d return=%.1f final_pos=%d  (press any key)", ep,
             ep_return, env.pos);
    nodelay(stdscr, FALSE);
    getch();
    nodelay(stdscr, TRUE);
  }

  SAM_MUZE_destroy(cortex);
  SAM_destroy(sam);
  endwin();
  return 0;
}
