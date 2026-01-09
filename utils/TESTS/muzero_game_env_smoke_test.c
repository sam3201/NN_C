// utils/TESTS/muze_curses_viz_test.c
// Terminal visualizer for MUZE using curses (ncurses).
//
// Build (macOS):
//   gcc muze_curses_viz_test.c <your_muze_objs...> -lncurses -o muze_curses_viz
//
// Notes:
// - You MUST fill in the ADAPTER section to match your APIs.
// - Everything else is generic and should stay as-is.

#include <curses.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ============================================================
// ADAPTER SECTION (EDIT THIS TO MATCH YOUR CODEBASE)
// ============================================================
//
// Goal: Provide a minimal interface for the visualizer.
//
// 1) "tick" function that advances the system once (self-play / train step /
// inference step) 2) a "stats" struct you can populate from your
// runtime/trainer/selfplay data 3) optional: a text framebuffer to render your
// env (grid/board)

typedef struct {
  // Fill with whatever you can easily query each frame:
  int episode;
  int step_in_episode;

  double last_reward;
  double value_estimate;
  double policy_entropy;

  int replay_size;
  int games_in_buffer;

  double loss_total;
  double loss_policy;
  double loss_value;
  double loss_reward;

  double lr;

  // search stats if you have them
  int mcts_sims;
  int mcts_depth;
} VizStats;

// Optional env render buffer (ASCII grid). If you don’t have a grid env,
// you can leave width/height = 0 and fb = NULL.
typedef struct {
  int width;
  int height;
  // fb is row-major, each cell one char (not null-terminated per row).
  const char *fb;
} VizFrame;

// ----- You implement these 3 functions -----
//
// Return true to keep running, false to stop (or you can stop via 'q' key too).
static bool muze_viz_tick(VizStats *out_stats) {
  // TODO: ADVANCE your system by one step:
  // Examples (depending on how your code is organized):
  //   - self_play_step(...)
  //   - trainer_step(...)
  //   - muze_loop_step(...)
  //   - runtime_step(...)
  //
  // Populate out_stats with whatever you have.

  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->episode = 0;
  out_stats->step_in_episode = 0;
  out_stats->last_reward = 0.0;
  out_stats->value_estimate = 0.0;
  out_stats->policy_entropy = 0.0;
  out_stats->replay_size = 0;
  out_stats->games_in_buffer = 0;
  out_stats->loss_total = 0.0;
  out_stats->loss_policy = 0.0;
  out_stats->loss_value = 0.0;
  out_stats->loss_reward = 0.0;
  out_stats->lr = 0.0;
  out_stats->mcts_sims = 0;
  out_stats->mcts_depth = 0;

  // Return true to keep running:
  return true;
}

static VizFrame muze_viz_get_frame(void) {
  // TODO: Return a pointer to a stable framebuffer of ASCII tiles.
  // If you don’t have a framebuffer, return {0,0,NULL}.
  VizFrame f = {0, 0, NULL};
  return f;
}

static void muze_viz_shutdown(void) {
  // TODO: Cleanup anything you initialized in your adapter.
  // (replay buffer, env, model, threads, etc.)
}

// ============================================================
// VISUALIZER (no edits usually needed below)
// ============================================================

static uint64_t now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000ULL);
}

static void draw_box(int y, int x, int h, int w, const char *title) {
  mvaddch(y, x, ACS_ULCORNER);
  mvhline(y, x + 1, ACS_HLINE, w - 2);
  mvaddch(y, x + w - 1, ACS_URCORNER);

  mvvline(y + 1, x, ACS_VLINE, h - 2);
  mvvline(y + 1, x + w - 1, ACS_VLINE, h - 2);

  mvaddch(y + h - 1, x, ACS_LLCORNER);
  mvhline(y + h - 1, x + 1, ACS_HLINE, w - 2);
  mvaddch(y + h - 1, x + w - 1, ACS_LRCORNER);

  if (title && title[0]) {
    mvprintw(y, x + 2, " %s ", title);
  }
}

static void draw_hud(int y, int x, int w, const VizStats *s, double fps,
                     bool paused) {
  char pbuf[32];
  snprintf(pbuf, sizeof(pbuf), "%s", paused ? "PAUSED" : "RUNNING");

  mvprintw(y + 0, x + 0, "Status: %-7s   FPS: %6.1f", pbuf, fps);
  mvprintw(y + 1, x + 0, "Episode: %-6d   Step: %-6d   Reward: %+8.4f",
           s->episode, s->step_in_episode, s->last_reward);
  mvprintw(y + 2, x + 0, "Value:   %+8.4f   Entropy: %8.4f   LR: %g",
           s->value_estimate, s->policy_entropy, s->lr);
  mvprintw(y + 3, x + 0, "Replay:  size=%-7d games=%-6d", s->replay_size,
           s->games_in_buffer);
  mvprintw(y + 4, x + 0,
           "Loss:    total=%8.5f  policy=%8.5f  value=%8.5f  reward=%8.5f",
           s->loss_total, s->loss_policy, s->loss_value, s->loss_reward);
  mvprintw(y + 5, x + 0, "MCTS:    sims=%-6d depth=%-4d", s->mcts_sims,
           s->mcts_depth);

  // Controls
  mvprintw(y + 7, x + 0,
           "Keys:  q=quit   p=pause   n=step(when paused)   r=reset screen");
  mvprintw(y + 8, x + 0,
           "       +/- = speed    (speed affects tick rate / sleep)");
  // clear remaining line space if needed
  (void)w;
}

static void draw_frame(int y, int x, int max_h, int max_w, VizFrame f) {
  if (f.width <= 0 || f.height <= 0 || !f.fb) {
    mvprintw(y, x, "(no framebuffer provided by adapter)");
    return;
  }

  int draw_h = (f.height < max_h) ? f.height : max_h;
  int draw_w = (f.width < max_w) ? f.width : max_w;

  for (int r = 0; r < draw_h; r++) {
    for (int c = 0; c < draw_w; c++) {
      char ch = f.fb[r * f.width + c];
      mvaddch(y + r, x + c, (ch == 0) ? ' ' : ch);
    }
  }
}

int main(void) {
  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);
  nodelay(stdscr, TRUE);
  curs_set(0);

  // If your terminal supports colors, you can enable them:
  if (has_colors()) {
    start_color();
    use_default_colors();
  }

  bool running = true;
  bool paused = false;

  // Speed control (microseconds sleep after each tick)
  int64_t sleep_us = 0; // start fast
  const int64_t sleep_min = 0;
  const int64_t sleep_max = 50000;

  VizStats stats;
  memset(&stats, 0, sizeof(stats));

  uint64_t last_fps_t = now_us();
  uint64_t frames = 0;
  double fps = 0.0;

  while (running) {
    int ch = getch();
    if (ch != ERR) {
      if (ch == 'q' || ch == 'Q')
        running = false;
      else if (ch == 'p' || ch == 'P')
        paused = !paused;
      else if (ch == 'r' || ch == 'R') {
        clear();
        refresh();
      } else if (ch == '+') {
        sleep_us -= 2000;
        if (sleep_us < sleep_min)
          sleep_us = sleep_min;
      } else if (ch == '-') {
        sleep_us += 2000;
        if (sleep_us > sleep_max)
          sleep_us = sleep_max;
      } else if (ch == 'n' || ch == 'N') {
        // single-step only when paused
        if (paused) {
          (void)muze_viz_tick(&stats);
        }
      }
    }

    if (!paused) {
      if (!muze_viz_tick(&stats)) {
        running = false;
      }
    }

    // Layout
    int term_h, term_w;
    getmaxyx(stdscr, term_h, term_w);

    int hud_h = 12;
    int hud_w = term_w - 2;
    int frame_y = hud_h + 2;
    int frame_h = term_h - frame_y - 2;
    int frame_w = term_w - 2;

    erase();

    draw_box(0, 0, hud_h, term_w, "MUZE Visualizer");
    draw_hud(1, 2, hud_w, &stats, fps, paused);

    draw_box(hud_h, 0, term_h - hud_h, term_w, "Environment / Frame");
    VizFrame f = muze_viz_get_frame();
    draw_frame(frame_y, 2, frame_h, frame_w, f);

    // FPS calc
    frames++;
    uint64_t t = now_us();
    if (t - last_fps_t >= 500000) { // update every 0.5s
      fps = (double)frames * 1000000.0 / (double)(t - last_fps_t);
      frames = 0;
      last_fps_t = t;
    }

    refresh();

    if (sleep_us > 0)
      usleep((useconds_t)sleep_us);
  }

  endwin();
  muze_viz_shutdown();
  return 0;
}
