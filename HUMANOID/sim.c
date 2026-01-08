#include "../SAM/SAM.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =======================
// CONFIG
// =======================
#define FPS 60.0f
#define EPISODE_SECONDS 8.0f

// Grid observation: OBS_DIM must equal GRID_W * GRID_H
#define GRID_W SCREEN_WIDTH
#define GRID_H SCREEN_HEIGHT
#if (OBS_DIM != (GRID_W * GRID_H))
#error "OBS_DIM must equal GRID_W*GRID_H"
#endif

// Cell IDs (floats are fine)
#define CELL_EMPTY 0.0f
#define CELL_SELF 1.0f
#define CELL_OTHER 2.0f
#define CELL_PLAT 3.0f
// ------------ Observation Grid (runtime) ------------
// Choose a scale. 1280x800 with 20px tiles => 64x40 grid.
#define TILE_PX 20

static int GRID_W = 0;
static int GRID_H = 0;
static int OBS_DIM = 0;

// Cell IDs
#define CELL_EMPTY 0.0f
#define CELL_SELF 1.0f
#define CELL_OTHER 2.0f
#define CELL_PLAT 3.0f

// World-space view window around the agent (in pixels)
#define VIEW_W_PX 720.0f
#define VIEW_H_PX 560.0f

// Anti-stuck shaping
// if no new best_alt for this long -> punish / reset
#define STUCK_NO_PROGRESS_SECS 1.5f
#define STUCK_RESET_SECS 3.0f // hard reset if really stuck
#define STUCK_PENALTY 0.02f   // per second penalty when stuck (soft shaping)

// Upward-velocity shaping (airborne only)
// Reward small positive signal when moving upward (vel.y < 0)
#define UP_VEL_SCALE                                                           \
  1800.0f // normalize denominator (matches your obs vy scale)
#define UP_VEL_REWARD 0.0030f // per-second max bonus (tiny)

#define STILL_EPS_PX 1.5f // how close counts as "not moving"
#define STILL_SECS 1.0f   // time with tiny movement before we consider "still"

#define WORKER_COUNT 4
#define MAX_AGENTS 8

#define FIXED_DT (1.0f / 120.0f)
#define MAX_ACCUM_DT (0.25f)

// =======================
// JUMP KING ENV
// =======================
#define PLATFORM_MAX 256
#define WORLD_HEIGHT 8000.0f
#define GRAVITY_Y 2600.0f
#define MOVE_SPEED 260.0f
#define AIR_CONTROL 0.35f

#define JUMP_CHARGE_RATE 1.6f
#define JUMP_CHARGE_MAX 1.0f
#define JUMP_VY_MIN -650.0f
#define JUMP_VY_MAX -1550.0f
#define JUMP_VX_MAX 520.0f

typedef struct {
  float x, y, w, h;
  int one_way; // 1 = one-way from below
} Platform;

typedef struct {
  Vector2 pos;
  Vector2 vel;
  float radius;

  int on_ground;
  float prev_y;

  int charging;
  float charge; // 0..1
} PlayerJK;

static Platform g_plats[PLATFORM_MAX];
static int g_plat_count = 0;

enum {
  ACT_NONE = 0,
  ACT_LEFT,
  ACT_RIGHT,
  ACT_CHARGE,
  ACT_RELEASE,
  ACTION_COUNT
};

/*
typedef struct {
  float obs[OBS_DIM];
  int n;
} ObsFixed;
*/

typedef struct {
  float *obs; // heap buffer, length = OBS_DIM
  int n;      // should end up == OBS_DIM
} ObsDyn;

typedef struct {
  bool alive;
  Vector2 spawn_origin;

  PlayerJK pl;

  float episode_time;
  float episode_limit;

  float best_alt; // best height this episode (higher is better)

  // --- SAM/MUZE ---
  SAM_t *sam;
  MuCortex *cortex;
  int last_action;

  // fixed step + control rate
  float accum_dt;
  float control_timer;
  float control_period;

  float time_since_progress; // seconds since best_alt improved
  float last_best_alt;       // best_alt snapshot for progress detection

  Vector2 last_pos;     // last position sample
  float pos_still_time; // seconds with tiny movement

  float pending_reward;
  ObsDyn last_obs;
  int has_last_transition;

  float reward_accumulator;

} Agent;

// =======================
// GLOBALS
// =======================
// static int SCREEN_WIDTH = 1280;
// static int SCREEN_HEIGHT = 800;
static float g_dt = 1.0f / FPS;

pthread_mutex_t job_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t job_cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;
bool job_quit = 0;
bool job_active = 0;
unsigned int job_next_agent = 0;
size_t job_done_workers = 0;
pthread_t workers[WORKER_COUNT];

Agent agents[MAX_AGENTS];

// =======================
// MATH / HELPERS
// =======================
static inline float clampf(float x, float a, float b) {
  return x < a ? a : (x > b ? b : x);
}

static Vector2 clamp_target_to_world(Vector2 t) {
  // Horizontal world bounds (you already clamp player 0..SCREEN_WIDTH)
  if (t.x < 0)
    t.x = 0;
  if (t.x > (float)SCREEN_WIDTH)
    t.x = (float)SCREEN_WIDTH;

  // Vertical bounds: keep target within the tower range
  // ground is around SCREEN_HEIGHT-40, top is about -WORLD_HEIGHT.
  float minY = -WORLD_HEIGHT;                // highest visible world Y
  float maxY = (float)SCREEN_HEIGHT - 40.0f; // ground
  if (t.y < minY)
    t.y = minY;
  if (t.y > maxY)
    t.y = maxY;

  return t;
}

static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static inline int grid_index(int gx, int gy) { return gy * GRID_W + gx; }

static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline void world_to_grid(const Vector2 origin, float x, float y,
                                 int *out_gx, int *out_gy) {
  float u = (x - origin.x) / VIEW_W_PX; // 0..1
  float v = (y - origin.y) / VIEW_H_PX; // 0..1
  int gx = (int)floorf(u * (float)GRID_W);
  int gy = (int)floorf(v * (float)GRID_H);
  *out_gx = gx;
  *out_gy = gy;
}

// Stamp an AABB (platform) into grid cells
static void stamp_rect(float *grid, Vector2 origin, float rx, float ry,
                       float rw, float rh, float value) {
  int gx0, gy0, gx1, gy1;
  world_to_grid(origin, rx, ry, &gx0, &gy0);
  world_to_grid(origin, rx + rw, ry + rh, &gx1, &gy1);

  // Expand slightly so thin platforms still appear
  gx0 = clampi(gx0 - 1, 0, GRID_W - 1);
  gy0 = clampi(gy0 - 1, 0, GRID_H - 1);
  gx1 = clampi(gx1 + 1, 0, GRID_W - 1);
  gy1 = clampi(gy1 + 1, 0, GRID_H - 1);

  for (int gy = gy0; gy <= gy1; gy++) {
    for (int gx = gx0; gx <= gx1; gx++) {
      int idx = grid_index(gx, gy);
      if (grid[idx] < value)
        grid[idx] = value; // keep higher priority
    }
  }
}

// Stamp a point/circle-ish (agent) into a single cell (or small radius if
// desired)
static void stamp_point(float *grid, Vector2 origin, float x, float y,
                        float value) {
  int gx, gy;
  world_to_grid(origin, x, y, &gx, &gy);
  if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H)
    return;
  int idx = grid_index(gx, gy);
  if (grid[idx] < value)
    grid[idx] = value;
}

static void obs_alloc(ObsDyn *o) {
  o->n = 0;
  o->obs = (float *)calloc((size_t)OBS_DIM, sizeof(float));
}

static void obs_free(ObsDyn *o) {
  free(o->obs);
  o->obs = NULL;
  o->n = 0;
}

static int circle_overlaps_rect(Vector2 c, float r, const Platform *pl) {
  float cx = clampf(c.x, pl->x, pl->x + pl->w);
  float cy = clampf(c.y, pl->y, pl->y + pl->h);
  float dx = c.x - cx;
  float dy = c.y - cy;
  return (dx * dx + dy * dy) <= r * r;
}

// =======================
// PLATFORMS
// =======================
static void build_platforms(void) {
  g_plat_count = 0;

  // ground
  g_plats[g_plat_count++] = (Platform){
      .x = -2000,
      .y = (float)SCREEN_HEIGHT - 40,
      .w = 5000,
      .h = 40,
      .one_way = 0,
  };

  float y = (float)SCREEN_HEIGHT - 120.0f;
  float x = 200.0f;

  for (int i = 0; i < PLATFORM_MAX - 1; i++) {
    float w = 120 + (rand() % 120);
    float h = 14;
    float dx = (rand() % 240) - 120;
    float dy = 60 + (rand() % 60); // 60..119

    x = clampf(x + dx, 80.0f, (float)SCREEN_WIDTH - 200.0f);
    y -= dy;

    g_plats[g_plat_count++] = (Platform){
        .x = x,
        .y = y,
        .w = w,
        .h = h,
        .one_way = 1,
    };

    if (y < -WORLD_HEIGHT)
      break;
  }
}

static void solve_player_platforms(PlayerJK *p) {
  p->on_ground = 0;

  for (int i = 0; i < g_plat_count; i++) {
    Platform *pl = &g_plats[i];

    if (pl->one_way) {
      if (p->vel.y < 0)
        continue; // rising: pass through

      float top = pl->y;
      float prev_bottom = p->prev_y + p->radius;
      float cur_bottom = p->pos.y + p->radius;

      // must cross top surface downward
      if (!(prev_bottom <= top && cur_bottom >= top))
        continue;

      // must be horizontally over platform
      if (p->pos.x < pl->x - p->radius || p->pos.x > pl->x + pl->w + p->radius)
        continue;

      p->pos.y = top - p->radius;
      p->vel.y = 0;
      p->on_ground = 1;
      continue;
    }

    // solid (ground)
    if (circle_overlaps_rect(p->pos, p->radius, pl)) {
      p->pos.y = pl->y - p->radius;
      p->vel.y = 0;
      p->on_ground = 1;
    }
  }
}

// =======================
// OBS
// =======================
static void encode_observation_jk(const Agent *a, ObsFixed *out) {
  out->n = 0;

  // Clear grid
  for (int i = 0; i < OBS_DIM; i++)
    out->obs[i] = CELL_EMPTY;

  const PlayerJK *p = &a->pl;

  // View window top-left in world space (centered on agent)
  Vector2 origin = {p->pos.x - VIEW_W_PX * 0.5f, p->pos.y - VIEW_H_PX * 0.5f};

  // 1) Stamp platforms
  for (int i = 0; i < g_plat_count; i++) {
    const Platform *pl = &g_plats[i];
    stamp_rect(out->obs, origin, pl->x, pl->y, pl->w, pl->h, CELL_PLAT);
  }

  // 2) Stamp other agents first (lower priority than self)
  for (int i = 0; i < MAX_AGENTS; i++) {
    const Agent *b = &agents[i];
    if (!b->alive)
      continue;
    if (b == a)
      continue;
    stamp_point(out->obs, origin, b->pl.pos.x, b->pl.pos.y, CELL_OTHER);
  }

  // 3) Stamp self last (highest priority)
  stamp_point(out->obs, origin, p->pos.x, p->pos.y, CELL_SELF);

  out->n = OBS_DIM;
}

// =======================
// AGENTS
// =======================
static void reset_agent_episode(Agent *a) {
  if (a->cortex && a->has_last_transition) {
    a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                     a->last_action, a->pending_reward, 1);
  }

  a->alive = true;
  a->episode_time = 0.0f;
  a->time_since_progress = 0.0f;
  a->last_best_alt = 0.0f;

  a->pos_still_time = 0.0f;

  a->pending_reward = 0.0f;
  a->has_last_transition = 0;
  memset(&a->last_obs, 0, sizeof(a->last_obs));
  a->reward_accumulator = 0.0f;

  a->pl.pos = a->spawn_origin;
  a->pl.vel = (Vector2){0, 0};
  a->pl.radius = 10.0f;
  a->pl.on_ground = 1;
  a->pl.charging = 0;
  a->pl.charge = 0.0f;
  a->pl.prev_y = a->pl.pos.y;
  a->last_pos = a->pl.pos;

  a->best_alt = 0.0f;
}

static void init_agents(void) {
  MuConfig cfg = {
      .obs_dim = OBS_DIM, .latent_dim = 64, .action_count = ACTION_COUNT};

  int cols = 16;
  float spacing = 70.0f;

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    memset(a, 0, sizeof(*a));

    int cx = i % cols;
    int cy = i / cols;
    Vector2 origin = {SCREEN_WIDTH * 0.2f + cx * spacing,
                      SCREEN_HEIGHT * 0.35f + cy * spacing * 1.1f};

    a->spawn_origin = origin;
    a->episode_limit = EPISODE_SECONDS;

    a->accum_dt = 0.0f;
    a->control_period = 1.0f / 30.0f;
    a->control_timer = 0.0f;
    a->last_action = ACT_NONE;

    a->sam = SAM_init(cfg.obs_dim, cfg.action_count, 4, 0);
    a->cortex = SAM_as_MUZE(a->sam);
    if (a->cortex) {
      a->cortex->policy_epsilon = 0.10f;
      a->cortex->policy_temperature = 1.0f;
      a->cortex->use_mcts = false;
    }

    reset_agent_episode(a);
  }
}

static void update_agent(Agent *a) {
  float dt = clampf(g_dt, 0.0f, MAX_ACCUM_DT);
  a->accum_dt += dt;
  a->episode_time += dt;

  // CONTROL TICK
  a->control_timer -= dt;
  if (a->control_timer <= 0.0f) {
    a->control_timer +=
        (a->control_period > 0) ? a->control_period : (1.0f / 20.0f);

    if (a->cortex && a->has_last_transition) {
      a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                       a->last_action, a->pending_reward, 0);
    }
    a->pending_reward = 0.0f;

    encode_observation_jk(a, &a->last_obs);
    a->last_action = a->cortex
                         ? muze_plan(a->cortex, a->last_obs.obs,
                                     (size_t)OBS_DIM, (size_t)ACTION_COUNT)
                         : ACT_NONE;

    a->has_last_transition = 1;
  }

  // PHYSICS FIXED STEP
  while (a->accum_dt >= FIXED_DT) {
    a->accum_dt -= FIXED_DT;

    PlayerJK *p = &a->pl;
    p->prev_y = p->pos.y;

    // horizontal intent
    float ax = 0.0f;
    if (a->last_action == ACT_LEFT)
      ax = -MOVE_SPEED;
    if (a->last_action == ACT_RIGHT)
      ax = +MOVE_SPEED;

    float control = p->on_ground ? 1.0f : AIR_CONTROL;
    p->vel.x += ax * control * FIXED_DT;

    // drag
    p->vel.x *= p->on_ground ? 0.92f : 0.985f;

    // --- charging state machine ---
    // If agent chooses CHARGE while grounded, start/continue charging.
    if (p->on_ground && a->last_action == ACT_CHARGE) {
      p->charging = 1;
    }

    // If charging, keep accumulating charge automatically (no need to keep
    // selecting CHARGE).
    if (p->charging && p->on_ground) {
      p->charge = clampf(p->charge + JUMP_CHARGE_RATE * FIXED_DT, 0.0f,
                         JUMP_CHARGE_MAX);
    }

    // Release jump only if currently charging and grounded
    if (p->on_ground && p->charging && a->last_action == ACT_RELEASE) {
      float t = clampf(p->charge, 0.0f, 1.0f);
      float vy = JUMP_VY_MIN + (JUMP_VY_MAX - JUMP_VY_MIN) * t;
      float vx = clampf(p->vel.x, -JUMP_VX_MAX, JUMP_VX_MAX);

      p->vel.y = vy;
      p->vel.x = vx;

      p->on_ground = 0;
      p->charging = 0;
      p->charge = 0.0f;
    }

    // If grounded and NOT charging, keep charge at 0 (prevents leftover charge)
    if (p->on_ground && !p->charging && a->last_action != ACT_CHARGE) {
      p->charge = 0.0f;
    }

    // gravity + integrate
    p->vel.y += GRAVITY_Y * FIXED_DT;
    p->pos.x += p->vel.x * FIXED_DT;
    p->pos.y += p->vel.y * FIXED_DT;

    // bounds
    if (p->pos.x < 0) {
      p->pos.x = 0;
      p->vel.x = 0;
    }
    if (p->pos.x > SCREEN_WIDTH) {
      p->pos.x = (float)SCREEN_WIDTH;
      p->vel.x = 0;
    }

    // collisions
    solve_player_platforms(p);

    // If we are not grounded, we cannot be charging.
    if (!p->on_ground) {
      p->charging = 0;
      // Optional: also clear charge so obs doesn't carry stale values.
      p->charge = 0.0f;
    }

    // --------------------------
    // Upward velocity shaping (airborne)
    // --------------------------
    if (!p->on_ground) {
      // upward speed is -vel.y (since y-down)
      float up = (-p->vel.y) / UP_VEL_SCALE; // roughly 0..~1
      up = clampf(up, 0.0f, 1.0f);
      a->pending_reward += UP_VEL_REWARD * up * FIXED_DT;
    }

    // reward: maximize height
    float groundY = (float)SCREEN_HEIGHT - 40.0f;
    float alt = groundY - p->pos.y;
    if (alt > a->best_alt) {
      float delta = alt - a->best_alt;
      a->best_alt = alt;
      a->pending_reward += 0.02f * (delta / 50.0f);
    }

    a->pending_reward += 0.001f; // alive drip

    // --------------------------
    // Anti-stuck reward shaping
    // --------------------------

    // 1) "Progress" = improving best_alt
    if (a->best_alt > a->last_best_alt + 0.001f) {
      a->last_best_alt = a->best_alt;
      a->time_since_progress = 0.0f;
    } else {
      a->time_since_progress += FIXED_DT;
    }

    // 2) "Stillness" = barely moving in position
    float dxp = p->pos.x - a->last_pos.x;
    float dyp = p->pos.y - a->last_pos.y;
    float d2 = dxp * dxp + dyp * dyp;

    if (p->on_ground && !p->charging && d2 < (STILL_EPS_PX * STILL_EPS_PX)) {
      a->pos_still_time += FIXED_DT;
    } else {
      a->pos_still_time = 0.0f;
      a->last_pos = p->pos;
    }

    // Soft penalty if we haven't made upward progress for a bit
    if (a->time_since_progress > STUCK_NO_PROGRESS_SECS) {
      a->pending_reward -= STUCK_PENALTY * FIXED_DT;
    }

    // Hard reset if REALLY stuck (no progress for long OR totally still)
    if (a->time_since_progress > STUCK_RESET_SECS ||
        a->pos_still_time > (STILL_SECS + 0.5f)) {
      a->pending_reward -= 0.2f; // terminal penalty so it *feels* bad
      a->alive = false;
      break;
    }

    if (p->pos.y > groundY + 200.0f) {
      a->pending_reward -= 0.5f;
      a->alive = false;
      break;
    }
  }

  // TERMINATE
  if (!a->alive || a->episode_time >= a->episode_limit) {
    if (a->cortex && a->has_last_transition) {
      a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                       a->last_action, a->pending_reward, 1);
    }
    reset_agent_episode(a);
  }
}

// =======================
// THREADING
// =======================
static void run_agent_jobs(void) {
  pthread_mutex_lock(&job_mtx);
  job_next_agent = 0;
  job_done_workers = 0;
  job_active = 1;

  pthread_cond_broadcast(&job_cv);

  while (job_active) {
    pthread_cond_wait(&done_cv, &job_mtx);
  }
  pthread_mutex_unlock(&job_mtx);
}

static void *agent_worker(void *arg) {
  (void)arg;

  for (;;) {
    pthread_mutex_lock(&job_mtx);
    while (!job_active && !job_quit) {
      pthread_cond_wait(&job_cv, &job_mtx);
    }
    if (job_quit) {
      pthread_mutex_unlock(&job_mtx);
      break;
    }
    pthread_mutex_unlock(&job_mtx);

    for (;;) {
      int idx;
      pthread_mutex_lock(&job_mtx);
      idx = (int)job_next_agent++;
      pthread_mutex_unlock(&job_mtx);

      if (idx >= MAX_AGENTS)
        break;
      update_agent(&agents[idx]);
    }

    pthread_mutex_lock(&job_mtx);
    job_done_workers++;
    if (job_done_workers >= WORKER_COUNT) {
      job_active = 0;
      pthread_cond_signal(&done_cv);
    }
    pthread_mutex_unlock(&job_mtx);
  }

  return NULL;
}

static void start_workers(void) {
  pthread_mutex_lock(&job_mtx);
  job_quit = 0;
  job_active = 0;
  job_next_agent = 0;
  job_done_workers = 0;
  pthread_mutex_unlock(&job_mtx);

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_create(&workers[i], NULL, agent_worker, NULL);
  }
}

static void stop_workers(void) {
  pthread_mutex_lock(&job_mtx);
  job_quit = 1;
  pthread_cond_broadcast(&job_cv);
  pthread_mutex_unlock(&job_mtx);

  for (int i = 0; i < WORKER_COUNT; i++) {
    pthread_join(workers[i], NULL);
  }
}

// =======================
// RENDERING
// =======================
static void draw_platforms(float camY) {
  for (int i = 0; i < g_plat_count; i++) {
    Platform *pl = &g_plats[i];
    DrawRectangle((int)pl->x, (int)(pl->y + camY), (int)pl->w, (int)pl->h,
                  pl->one_way ? GRAY : DARKGRAY);
  }
}

static void draw_agent_jk(const Agent *a, float camY, bool highlight) {
  Vector2 p = a->pl.pos;
  int x = (int)p.x;
  int y = (int)(p.y + camY);
  int r = (int)a->pl.radius;

  if (highlight) {
    DrawCircle(x, y, r + 6, GOLD);
    DrawCircle(x, y, r + 3, ORANGE);
    DrawCircle(x, y, r, WHITE);
  } else {
    DrawCircle(x, y, r, RAYWHITE);
  }
}

static int find_best_agent_index(void) {
  int best = 0;
  float best_alt = agents[0].best_alt;

  for (int i = 1; i < MAX_AGENTS; i++) {
    if (agents[i].best_alt > best_alt) {
      best_alt = agents[i].best_alt;
      best = i;
    }
  }
  return best;
}

// =======================
// MAIN
// =======================
int main(void) {
  srand((unsigned int)time(NULL));

  InitWindow(1280, 800, "Jump King (MUZE)");
  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  SetTargetFPS((int)FPS);

  build_platforms();
  init_agents();
  start_workers();

  Camera2D cam = {0};
  cam.offset = (Vector2){SCREEN_WIDTH * 0.5f, SCREEN_HEIGHT * 0.5f};
  cam.target = (Vector2){SCREEN_WIDTH * 0.5f, SCREEN_HEIGHT * 0.5f};
  cam.rotation = 0.0f;
  cam.zoom = 1.0f;

  while (!WindowShouldClose()) {
    g_dt = GetFrameTime();

    // simulate
    run_agent_jobs();

    // pick best agent
    int best_i = find_best_agent_index();
    Agent *ba = &agents[best_i];

    // desired camera target = best agent position
    Vector2 desired = ba->pl.pos;
    desired = clamp_target_to_world(desired);

    // smooth camera so it doesn't snap (tune 8..20)
    float smooth = 12.0f;
    float t = 1.0f - expf(-smooth * g_dt);
    cam.target.x = lerpf(cam.target.x, desired.x, t);
    cam.target.y = lerpf(cam.target.y, desired.y, t);

    BeginDrawing();
    ClearBackground(BLACK);

    BeginMode2D(cam);

    // draw world in world coordinates (NO camY needed)
    for (int i = 0; i < g_plat_count; i++) {
      Platform *pl = &g_plats[i];
      DrawRectangle((int)pl->x, (int)pl->y, (int)pl->w, (int)pl->h,
                    pl->one_way ? GRAY : DARKGRAY);
    }

    for (int i = 0; i < MAX_AGENTS; i++) {
      Vector2 p = agents[i].pl.pos;
      int x = (int)p.x;
      int y = (int)p.y;
      int r = (int)agents[i].pl.radius;

      if (i == best_i) {
        DrawCircle(x, y, r + 6, GOLD);
        DrawCircle(x, y, r + 3, ORANGE);
        DrawCircle(x, y, r, WHITE);
      } else {
        DrawCircle(x, y, r, RAYWHITE);
      }
    }

    EndMode2D();

    // HUD (screen space) — must be AFTER EndMode2D, and inside BeginDrawing
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 20, 20, RAYWHITE);
    DrawText(TextFormat("Best agent: %d", best_i), 20, 45, 20, RAYWHITE);
    DrawText(TextFormat("Best alt: %.1f", ba->best_alt), 20, 70, 20, RAYWHITE);
    DrawText(TextFormat("Charge: %.2f", ba->pl.charge), 20, 95, 20, RAYWHITE);
    DrawText(TextFormat("Charging: %d", ba->pl.charging), 20, 120, 20,
             RAYWHITE);
    DrawText(TextFormat("OnGround: %d", ba->pl.on_ground), 20, 145, 20,
             RAYWHITE);
    DrawText(TextFormat("LastAct: %d", ba->last_action), 20, 170, 20, RAYWHITE);

    EndDrawing();
  }

  stop_workers();
  CloseWindow();
  return 0;
}
