#include "../../SAM/SAM.h"
#include "../NN/MUZE/all.h"
#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void draw_env_line(int y, int x, const ToyEnvState *env, int goal_pos) {
  move(y, x);
  for (int i = 0; i < (int)env->size; i++) {
    if (i == env->pos)
      addch('X');
    else if (i == goal_pos)
      addch('G');
    else
      addch('.');
  }
}

static void draw_obs(int y, int x, const float *obs, int n) {
  mvprintw(y, x, "obs: ");
  for (int i = 0; i < n; i++) {
    printw("%.0f", obs[i]);
    if (i + 1 < n)
      printw(" ");
  }
}

int main(void) {
  srand(1);

  ToyEnvState env = {.pos = 0, .size = 8};
  const size_t obs_dim = env.size;
  const int action_count = 2;
  const int goal_pos = (int)env.size - 1;

  initscr();
  cbreak();
  noecho();
  curs_set(0);
  keypad(stdscr, TRUE);
  nodelay(stdscr, TRUE); /* don't block on input */

  SAM_t *sam = SAM_init((size_t)obs_dim, (size_t)action_count, 2, 0);
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
  cortex->use_mcts = true;
  cortex->mcts_model = cortex->brain;
  cortex->mcts_params.num_simulations = 100; /* tweak */

  /* Optional: slow down */
  const int delay_ms = 35;

  const int episodes = 50;
  const int max_steps = 128;

  for (int ep = 0; ep < episodes; ep++) {
    float obs[obs_dim];
    toy_env_reset(&env, obs);

    float ep_return = 0.0f;

    for (int step = 0; step < max_steps; step++) {
      int ch = getch();
      if (ch == 'q' || ch == 'Q') {
        SAM_MUZE_destroy(cortex);
        SAM_destroy(sam);
        endwin();
        return 0;
      }

      int action = muze_select_action(cortex, obs, (size_t)obs_dim,
                                      (size_t)action_count);

      float next_obs[obs_dim];
      float env_reward = 0.0f;
      int env_done = 0;

      if (toy_env_step(&env, action, next_obs, &env_reward, &env_done) != 0) {
        mvprintw(0, 0, "env_step error\n");
        refresh();
        break;
      }

      float reward = (env.pos == goal_pos) ? 1.0f : 0.0f;
      int done = (env.pos == goal_pos) ? 1 : 0;

      cortex->learn(cortex->brain, obs, obs_dim, action, reward, done);

      ep_return += reward;
      memcpy(obs, next_obs, sizeof(obs));

      clear();
      mvprintw(0, 0, "Toy Muze Visualizer (press q to quit)");
      mvprintw(2, 0, "ep=%d step=%d action=%d reward=%.1f pos=%d done=%d", ep,
               step, action, reward, env.pos, done);

      draw_env_line(4, 0, &env, goal_pos);
      draw_obs(6, 0, obs, (int)obs_dim);

      mvprintw(8, 0, "episode_return=%.1f", ep_return);

      refresh();
      napms(delay_ms);

      if (done)
        break;
    }

    /* tiny pause between episodes */
    napms(120);
  }

  SAM_MUZE_destroy(cortex);
  SAM_destroy(sam);
  endwin();
  return 0;
}
