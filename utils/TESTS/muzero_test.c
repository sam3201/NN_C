#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"
#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // usleep

static void curses_print_obs(int row, int col, const float *obs, int n) {
  mvprintw(row, col, "obs=[");
  int x = col + 5;
  for (int i = 0; i < n; i++) {
    mvprintw(row, x, "%.0f", obs[i]);
    x += 1;
    if (i + 1 < n) {
      mvprintw(row, x, " ");
      x += 1;
    }
  }
  mvprintw(row, x, "]");
}

static void display_env_curses(int row, int col, const ToyEnvState *env,
                               int goal_pos) {
  for (int i = 0; i < env->size; i++) {
    char ch = ' ';
    if (i == env->pos)
      ch = 'X';
    else if (i == goal_pos)
      ch = 'G';
    mvaddch(row, col + i, ch);
  }
}

int main(void) {
  srand(1);

  ToyEnvState env = {.pos = 0, .size = 8};
  size_t obs_dim = (size_t)env.size;
  const int action_count = 2;
  const int goal_pos = env.size - 1;

  // ---- curses init (MUST be before curs_set) ----
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  keypad(stdscr, TRUE);
  nodelay(stdscr, TRUE); // non-blocking getch (optional)
  // timeout(0); // alternative

  SAM_t *sam = SAM_init(obs_dim, (size_t)action_count, 2, 0);
  if (!sam) {
    endwin();
    fprintf(stderr, "SAM_init failed\n");
    return 1;
  }

  MuCortex *cortex = SAM_as_MUZE(sam);
  if (!cortex) {
    endwin();
    fprintf(stderr, "SAM_as_MUZE failed\n");
    SAM_destroy(sam);
    return 1;
  }

  const int episodes = 50;
  const int max_steps = 128;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[8]; // env.size=8; keep it explicit for curses demo safety
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;

    for (int step = 0; step < max_steps; step++) {
      // (optional) allow quitting with 'q'
      int ch = getch();
      if (ch == 'q' || ch == 'Q') {
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
        erase();
        mvprintw(0, 0, "env_step error");
        refresh();
        break;
      }

      // Use env outputs directly (cleaner)
      float reward = env_reward;
      int done = env_done;

      // Learn from the transition *before* overwriting obs
      cortex->learn(cortex->brain, obs, obs_dim, action, reward, done);

      ep_return += reward;
      memcpy(obs, next_obs, obs_dim * sizeof(float));

      // ---- draw ----
      erase();
      mvprintw(0, 0, "SAM MUZE ToyEnv  (press q to quit)");
      mvprintw(1, 0,
               "ep=%d/%d  step=%d/%d  action=%d  reward=%.1f  return=%.1f", ep,
               episodes - 1, step, max_steps - 1, action, reward, ep_return);

      mvprintw(3, 0, "world: ");
      display_env_curses(3, 7, &env, goal_pos);

      mvprintw(5, 0, "pos=%d  goal=%d  done=%d", env.pos, goal_pos, done);
      curses_print_obs(7, 0, obs, (int)obs_dim);

      refresh();

      // control animation speed
      usleep(80 * 1000); // 80ms per step

      if (done)
        break;
    }

    // episode summary pause
    mvprintw(9, 0, "episode %d return=%.1f final_pos=%d   (continuing...)", ep,
             ep_return, env.pos);
    refresh();
    usleep(250 * 1000);
  }

  SAM_MUZE_destroy(cortex);
  SAM_destroy(sam);

  endwin();
  return 0;
}
