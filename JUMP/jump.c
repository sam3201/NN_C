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

#define WORKER_COUNT 4
#define MAX_AGENTS 8

// Fixed-step physics
#define FIXED_DT (1.0f / 120.0f)
#define MAX_ACCUM_DT 0.25f

// Jump King-ish world
#define WORLD_HEIGHT 8000.0f
#define GRAVITY_Y 2600.0f
#define MOVE_SPEED 260.0f
#define AIR_CONTROL 0.35f
#define AIR_ACCEL_SCALE 0.55f // extra drift accel amount (in air)
#define AIR_DRAG 0.992f       // keep velocity longer in air (was 0.985)

#define JUMP_CHARGE_RATE 1.6f
#define JUMP_CHARGE_MAX 1.0f
#define JUMP_VY_MIN -650.0f
#define JUMP_VY_MAX -1550.0f
#define JUMP_VX_MAX 520.0f

// Anti-stuck
#define STUCK_NO_PROGRESS_SECS 1.5f
#define STUCK_RESET_SECS 3.0f
#define STUCK_PENALTY 0.02f

#define STILL_EPS_PX 1.5f
#define STILL_SECS 1.0f

// Upward-velocity shaping
#define UP_VEL_SCALE 1800.0f
#define UP_VEL_REWARD 0.0030f

#define PLATFORM_MAX 256

#define CELL_EMPTY 0.0f
#define CELL_SELF 1.0f
#define CELL_OTHER 2.0f
#define CELL_ONEWAY 3.0f
#define CELL_GROUND 4.0f
#define CELL_LEDGE 5.0f // thin "top surface" highlight (helps massively)

#define PLATFORM_RAMP_MAX 32
#define RAMP_SLIDE_SCALE 0.45f // gravity component along ramp -> vx
#define CELL_RAMP 4.5f         // optional (between ground & ledge)

#define OBS_EXTRA 19

// ------------ Observation Grid (runtime) ------------
// 1280x800 with TILE_PX=20 => 64x40 grid (2560 cells)
#define TILE_PX 20

// World-space view window around the agent (in pixels)
#define VIEW_W_PX 720.0f
#define VIEW_H_PX 560.0f

#define EP_MAX_STEPS                                                           \
  4096 // enough for 8 seconds at 30Hz control: ~240, but give slack

static int SCREEN_WIDTH = 1280;
static int SCREEN_HEIGHT = 800;

static int GRID_W = 0;
static int GRID_H = 0;

static int OBS_GRID = 0; // GRID_W * GRID_H
static int OBS_DIM = 0;  // OBS_GRID + OBS_EXTRA

static MuModel *g_model = NULL;
static ReplayBuffer *g_rb = NULL;
static pthread_mutex_t g_rb_mtx = PTHREAD_MUTEX_INITIALIZER;

typedef enum { PLAT_RECT = 0, PLAT_RAMP = 1 } PlatKind;

typedef struct {
  PlatKind kind;
  int one_way;

  // rect
  float x, y, w, h;

  // ramp (line segment from (x,y) to (x2,y2))
  float x2, y2;
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
  int T;
  float *obs;    // [EP_MAX_STEPS * OBS_DIM]
  float *pi;     // [EP_MAX_STEPS * ACTION_COUNT]
  float *reward; // [EP_MAX_STEPS]
  int *action;   // [EP_MAX_STEPS]
  int *done;     // [EP_MAX_STEPS]
} EpisodeBuf;

typedef struct {
  bool alive;
  Vector2 spawn_origin;

  PlayerJK pl;

  float episode_time;
  float episode_limit;

  float best_alt; // best height this episode (higher is better)

  // decision / brain
  void *sam;        // or SAM* if you have a type
  MuCortex *cortex; // from MUZE/muze_cortex.h

  MCTSParams mcts_params; // if you want per-agent params

  int last_action; // ACT_*

  MuModel *model; // shared global
  EpisodeBuf ep;
  int ep_t;                    // current step index
  float last_pi[ACTION_COUNT]; // store last π for debugging if you want

  int episodes_done;

  // fixed step + control rate
  float accum_dt;
  float control_timer;
  float control_period;

  float time_since_progress; // seconds since best_alt improved
  float last_best_alt;       // best_alt snapshot for progress detection

  Vector2 last_pos;     // last position sample
  float pos_still_time; // seconds with tiny movement

  uint32_t rng;

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
static float agent_rand01(void *ctx) {
  Agent *a = (Agent *)ctx;
  return frand01(&a->rng);
}

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

static inline uint32_t xorshift32(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}
static inline float frand01(uint32_t *s) {
  return (xorshift32(s) >> 8) * (1.0f / 16777216.0f); // 24-bit mantissa
}
static inline int irand(uint32_t *s, int n) {
  return (int)(xorshift32(s) % (uint32_t)n);
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

static void stamp_disk(float *grid, Vector2 origin, float x, float y,
                       float value, int radius_cells) {
  int cx, cy;
  world_to_grid(origin, x, y, &cx, &cy);
  if (cx < 0 || cx >= GRID_W || cy < 0 || cy >= GRID_H)
    return;

  for (int dy = -radius_cells; dy <= radius_cells; dy++) {
    for (int dx = -radius_cells; dx <= radius_cells; dx++) {
      if (dx * dx + dy * dy > radius_cells * radius_cells)
        continue;
      int gx = cx + dx, gy = cy + dy;
      if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H)
        continue;
      int idx = grid_index(gx, gy);
      if (grid[idx] < value)
        grid[idx] = value;
    }
  }
}

static void stamp_platform_top(float *grid, Vector2 origin, float x, float y,
                               float w, float value) {
  // stamp a thin strip at y (platform top)
  stamp_rect(grid, origin, x, y - 2.0f, w, 4.0f, value);
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

static float epsilon_schedule(int ep) {
  // high exploration early, then decay
  float e0 = 0.80f;         // start
  float e1 = 0.05f;         // floor
  float half_life = 200.0f; // how fast it decays
  float t = (float)ep / half_life;
  float e = e1 + (e0 - e1) * expf(-t);
  return e;
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

static void episode_alloc(EpisodeBuf *ep) {
  ep->obs = (float *)malloc(sizeof(float) * EP_MAX_STEPS * OBS_DIM);
  ep->pi = (float *)malloc(sizeof(float) * EP_MAX_STEPS * ACTION_COUNT);
  ep->reward = (float *)malloc(sizeof(float) * EP_MAX_STEPS);
  ep->action = (int *)malloc(sizeof(int) * EP_MAX_STEPS);
  ep->done = (int *)malloc(sizeof(int) * EP_MAX_STEPS);
  ep->T = 0;
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

  // Simple ramps near spawn area
  g_plats[g_plat_count++] = (Platform){
      .kind = PLAT_RAMP,
      .one_way = 0,
      .x = 250.0f,
      .y = (float)SCREEN_HEIGHT - 160.0f,
      .x2 = 450.0f,
      .y2 = (float)SCREEN_HEIGHT - 100.0f, // down to right
  };

  g_plats[g_plat_count++] = (Platform){
      .kind = PLAT_RAMP,
      .one_way = 0,
      .x = 600.0f,
      .y = (float)SCREEN_HEIGHT - 120.0f,
      .x2 = 430.0f,
      .y2 = (float)SCREEN_HEIGHT - 70.0f, // down to left
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

    // ---- RAMP ----
    if (pl->kind == PLAT_RAMP) {
      // only land when falling
      if (p->vel.y < 0.0f)
        continue;

      float x0 = pl->x, y0 = pl->y;
      float x1 = pl->x2, y1 = pl->y2;

      float minx = fminf(x0, x1), maxx = fmaxf(x0, x1);
      if (p->pos.x < minx - p->radius || p->pos.x > maxx + p->radius)
        continue;

      // y on ramp at player's x (linear interpolation)
      float t =
          (fabsf(x1 - x0) > 0.0001f) ? ((p->pos.x - x0) / (x1 - x0)) : 0.0f;
      t = clampf(t, 0.0f, 1.0f);
      float y_on = y0 + (y1 - y0) * t;

      float prev_bottom = p->prev_y + p->radius;
      float cur_bottom = p->pos.y + p->radius;

      // must cross the ramp surface downward
      if (!(prev_bottom <= y_on && cur_bottom >= y_on))
        continue;

      // snap to ramp
      p->pos.y = y_on - p->radius;
      p->vel.y = 0.0f;
      p->on_ground = 1;

      // slide due to gravity component along slope
      float dx = (x1 - x0);
      float dy = (y1 - y0); // y-down
      float slope = (fabsf(dx) > 0.0001f) ? (dy / dx) : 0.0f;

      // sin(theta) where tan(theta)=slope -> slope/sqrt(1+slope^2)
      float sinth = slope / sqrtf(1.0f + slope * slope);

      // gravity pulls "down the ramp"
      p->vel.x += GRAVITY_Y * sinth * RAMP_SLIDE_SCALE * FIXED_DT;

      continue;
    }

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
static void encode_observation_jk(const Agent *a, ObsDyn *out) {
  out->n = 0;

  // Clear ONLY the grid portion
  for (int i = 0; i < OBS_GRID; i++)
    out->obs[i] = CELL_EMPTY;

  const PlayerJK *p = &a->pl;

  // View window centered on agent
  Vector2 origin = {p->pos.x - VIEW_W_PX * 0.5f, p->pos.y - VIEW_H_PX * 0.5f};

  // ----------------------------
  // PLATFORMS (ground vs one-way)
  // ----------------------------
  for (int i = 0; i < g_plat_count; i++) {
    const Platform *pl = &g_plats[i];

    if (pl->kind == PLAT_RECT) {
      float cell = pl->one_way ? CELL_ONEWAY : CELL_GROUND;
      stamp_rect(out->obs, origin, pl->x, pl->y, pl->w, pl->h, cell);
      stamp_platform_top(out->obs, origin, pl->x, pl->y, pl->w, CELL_LEDGE);
    } else {
      // Ramp: sample points along the segment and stamp
      const int S = 24;
      for (int s = 0; s <= S; s++) {
        float t = (float)s / (float)S;
        float rx = pl->x + (pl->x2 - pl->x) * t;
        float ry = pl->y + (pl->y2 - pl->y) * t;

        // “body” and “surface”
        stamp_rect(out->obs, origin, rx - 6.0f, ry - 6.0f, 12.0f, 12.0f,
                   CELL_GROUND);
        stamp_rect(out->obs, origin, rx - 6.0f, ry - 2.0f, 12.0f, 4.0f,
                   CELL_LEDGE);
      }
    }
  }

  // ----------------------------
  // AGENTS (stamp as disks)
  // ----------------------------
  // other agents first
  for (int i = 0; i < MAX_AGENTS; i++) {
    const Agent *b = &agents[i];
    if (!b->alive)
      continue;
    if (b == a)
      continue;
    stamp_disk(out->obs, origin, b->pl.pos.x, b->pl.pos.y, CELL_OTHER, 1);
  }

  // self last (highest priority)
  stamp_disk(out->obs, origin, p->pos.x, p->pos.y, CELL_SELF, 1);

  // ----------------------------
  // EXTRAS
  // ----------------------------
  int k = OBS_GRID;

  float groundY = (float)SCREEN_HEIGHT - 40.0f;
  float alt = groundY - p->pos.y;

  // normalized pose/vel
  float px = p->pos.x / (float)SCREEN_WIDTH;  // 0..1
  float py = p->pos.y / (float)SCREEN_HEIGHT; // can be negative
  float vx = clampf(p->vel.x / 800.0f, -2.0f, 2.0f);
  float vy = clampf(p->vel.y / 1800.0f, -2.0f, 2.0f);

  float alt_n = clampf(alt / 2000.0f, -2.0f, 2.0f);
  float stuck_n = a->time_since_progress / STUCK_RESET_SECS; // ~0..1+

  // tiny helper: "where is the nearest ledge above me?"
  // (purely from the known platforms, still within your sim; treat as extra
  // cue)
  float best_abs_dy = 1e30f;
  float best_dx = 0.0f;
  float best_dy = -600.0f; // default "somewhat above"

  for (int i = 0; i < g_plat_count; i++) {
    const Platform *pl = &g_plats[i];
    if (!pl->one_way)
      continue;
    if (pl->y >= p->pos.y)
      continue; // must be above

    float cx = pl->x + pl->w * 0.5f;
    float dx = cx - p->pos.x;
    float dy = pl->y - p->pos.y; // negative

    float abs_dy = -dy; // since dy is negative
    if (abs_dy < best_abs_dy) {
      best_abs_dy = abs_dy;
      best_dx = dx;
      best_dy = dy;
    }
  }

  float dx_n = clampf(best_dx / (float)SCREEN_WIDTH, -1.0f, 1.0f);
  float dy_n = clampf(best_dy / 600.0f, -3.0f, 0.0f);

  // extra useful scalars
  // 1) how fast the charge is growing *right now*
  float charge_rate_n =
      (p->on_ground && p->charging) ? (JUMP_CHARGE_RATE / 2.0f) : 0.0f;
  charge_rate_n = clampf(charge_rate_n, 0.0f, 1.0f);

  // 2) falling flag (y-down world): vel.y > 0 means falling
  float falling = (p->vel.y > 0.0f) ? 1.0f : 0.0f;

  // Movement intent (WASD-ish, derived from chosen action)
  float move_intent = 0.0f; // -1 left, +1 right
  float act_left = 0.0f, act_right = 0.0f, act_charge = 0.0f,
        act_release = 0.0f;

  if (a->last_action == ACT_LEFT) {
    move_intent = -1.0f;
    act_left = 1.0f;
  }
  if (a->last_action == ACT_RIGHT) {
    move_intent = +1.0f;
    act_right = 1.0f;
  }
  if (a->last_action == ACT_CHARGE)
    act_charge = 1.0f;
  if (a->last_action == ACT_RELEASE)
    act_release = 1.0f;

  // Fill OBS_EXTRA = 19
  // indices 0..18 (relative to k0=OBS_GRID)
  out->obs[k++] = px;                  // 0
  out->obs[k++] = py;                  // 1
  out->obs[k++] = vx;                  // 2
  out->obs[k++] = vy;                  // 3
  out->obs[k++] = (float)p->on_ground; // 4
  out->obs[k++] = (float)p->charging;  // 5
  out->obs[k++] = p->charge;           // 6
  out->obs[k++] = alt_n;               // 7
  out->obs[k++] = stuck_n;             // 8
  out->obs[k++] = dx_n;                // 9
  out->obs[k++] = dy_n;                // 10
  out->obs[k++] = charge_rate_n;       // 11
  out->obs[k++] = falling;             // 12

  // NEW movement obs
  out->obs[k++] = move_intent; // 13  (-1..1)
  out->obs[k++] = act_left;    // 14
  out->obs[k++] = act_right;   // 15
  out->obs[k++] = act_charge;  // 16
  out->obs[k++] = act_release; // 17

  out->obs[k++] = 1.0f; // 18 bias

  while (k < OBS_DIM)
    out->obs[k++] = 0.0f;
  out->n = OBS_DIM;
}

// =======================
// AGENTS
// =======================
static void reset_agent_episode(Agent *a) {
  // terminal learn (done=1) if we had a transition
  if (a->cortex && a->has_last_transition) {
    a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                     a->last_action, a->pending_reward, 1);
  }

  // Set exploration params for the episode we are ABOUT to start
  if (a->cortex) {
    a->cortex->policy_epsilon =
        epsilon_schedule(a->episodes_done); // ep0 -> 0.80
    a->cortex->policy_temperature =
        1.0f; // you can decay this too later if you want
  }

  // Now count this new episode as started
  a->episodes_done++;

  a->alive = true;
  a->episode_time = 0.0f;

  a->time_since_progress = 0.0f;
  a->last_best_alt = 0.0f;
  a->pos_still_time = 0.0f;

  a->pending_reward = 0.0f;
  a->has_last_transition = 0;

  a->pl.pos = a->spawn_origin;
  a->pl.vel = (Vector2){0, 0};
  a->pl.radius = 10.0f;
  a->pl.on_ground = 1;
  a->pl.charging = 0;
  a->pl.charge = 0.0f;
  a->pl.prev_y = a->pl.pos.y;

  a->last_pos = a->pl.pos;
  a->best_alt = 0.0f;

  // refresh obs for the new starting state
  encode_observation_jk(a, &a->last_obs);
  a->last_action = ACT_NONE;
}

static void init_agents(void) {
  MuConfig cfg = {
      .obs_dim = OBS_DIM, .latent_dim = 64, .action_count = ACTION_COUNT};
  g_model = mu_model_create(&cfg);
  g_rb = rb_create(200000, OBS_DIM, ACTION_COUNT); // big-ish

  int cols = 16;
  float spacing = 70.0f;

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    memset(a, 0, sizeof(*a));
    obs_alloc(&a->last_obs);

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

    uint32_t base = (uint32_t)time(NULL);
    a->rng = base ^ (0x9e3779b9u * (uint32_t)(i + 1));

    a->ep = NULL;
    episode_alloc();
    a->episodes_done = 0; // IMPORTANT: define episode 0 explicitly

    a->sam = SAM_init(cfg.obs_dim, cfg.action_count, 4, 0);
    a->cortex = SAM_as_MUZE(a->sam);
    if (a->cortex) {
      a->cortex->use_mcts = true;
      a->cortex->mcts_model = g_model;
      a->cortex->mcts_params.num_simulations = 80;
      a->cortex->mcts_params.max_depth = 16;
      a->cortex->mcts_params.discount = 0.997f;
      a->cortex->mcts_params.c_puct = 1.25f;
      a->cortex->mcts_params.temperature = 1.0f;
      a->cortex->mcts_params.dirichlet_alpha = 0.3f;
      a->cortex->mcts_params.dirichlet_eps = 0.25f;

      a->cortex->policy_temperature = 1.0f;
      a->cortex->policy_epsilon = epsilon_schedule(0);
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

    MCTSRng rng = {.ctx = a, .rand01 = agent_rand01};
    MCTSParams mp = a->mcts_params; // store in agent or global
    MCTSResult mr = mcts_run(g_model, a->last_obs.obs, &mp, &rng);

    // sample action from mr.pi using agent rng
    float r = frand01(&a->rng);
    float cum = 0.f;
    int chosen = 0;
    for (int i = 0; i < ACTION_COUNT; i++) {
      cum += mr.pi[i];
      if (r <= cum) {
        chosen = i;
        break;
      }
    }

    if (a->ep_t > 0) {
      a->ep.reward[a->ep_t - 1] = a->pending_reward;
    }
    a->pending_reward = 0.f;

    // now store current obs and π
    memcpy(a->ep.obs + a->ep_t * OBS_DIM, a->last_obs.obs,
           sizeof(float) * OBS_DIM);
    memcpy(a->ep.pi + a->ep_t * ACTION_COUNT, mr.pi,
           sizeof(float) * ACTION_COUNT);
    a->ep.action[a->ep_t] = chosen;
    a->ep.done[a->ep_t] = 0;
    a->ep_t++;

    int T = a->ep_t;
    if (T > 0) {
      a->ep.reward[T - 1] = a->pending_reward;
      a->ep.done[T - 1] = 1;

      // compute returns
      float *z = malloc(sizeof(float) * T);
      for (int t = 0; t < T; t++) {
        float acc = 0.f, g = 1.f;
        for (int k = t; k < T; k++) {
          acc += g * a->ep.reward[k];
          g *= 0.997f;
        }
        z[t] = acc;
      }

      pthread_mutex_lock(&g_rb_mtx);
      for (int t = 0; t < T; t++) {
        float *obs_t = a->ep.obs + t * OBS_DIM;
        float *pi_t = a->ep.pi + t * ACTION_COUNT;

        rb_push(g_rb, obs_t, pi_t, z[t]);

        // transition for dynamics (need next_obs)
        if (t + 1 < T) {
          float *obs_tp1 = a->ep.obs + (t + 1) * OBS_DIM;
          rb_push_transition(g_rb, obs_t, a->ep.action[t], a->ep.reward[t],
                             obs_tp1, a->ep.done[t]);
        }
      }
      pthread_mutex_unlock(&g_rb_mtx);

      free(z);
    }
    a->ep_t = 0;
    a->pending_reward = 0.f;

    if (a->cortex && a->has_last_transition) {
      a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                       a->last_action, a->pending_reward, 0);
    }
    a->pending_reward = 0.0f;

    encode_observation_jk(a, &a->last_obs);
    a->last_action = chosen;
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

    // Extra “drift” in air (Jump King-ish): small but persistent influence
    if (!p->on_ground &&
        (a->last_action == ACT_LEFT || a->last_action == ACT_RIGHT)) {
      p->vel.x += ax * (AIR_CONTROL * AIR_ACCEL_SCALE) * FIXED_DT;
    }

    // drag
    p->vel.x *= p->on_ground ? 0.92f : AIR_DRAG;

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

  // Choose grid size based on screen resolution
  GRID_W = SCREEN_WIDTH / TILE_PX;
  GRID_H = SCREEN_HEIGHT / TILE_PX;

  if (GRID_W < 8)
    GRID_W = 8;
  if (GRID_H < 6)
    GRID_H = 6;

  OBS_GRID = GRID_W * GRID_H;
  OBS_DIM = OBS_GRID + OBS_EXTRA;

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

    static int train_counter = 0;
    train_counter++;
    if ((train_counter % 2) == 0) { // every other frame, tune
      TrainerConfig tc = {
          .batch_size = 64,
          .train_steps = 50,
          .min_replay_size = 2048,
          .lr = 0.01f,
      };

      pthread_mutex_lock(&g_rb_mtx);
      trainer_train_from_replay(g_model, g_rb, &tc);
      trainer_train_dynamics(g_model, g_rb, &tc);
      pthread_mutex_unlock(&g_rb_mtx);
    }

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
    DrawText(TextFormat("Eps: %.3f  Ep: %d",
                        ba->cortex ? ba->cortex->policy_epsilon : 0.0f,
                        ba->episodes_done),
             20, 195, 20, RAYWHITE);

    EndDrawing();
  }

  for (int i = 0; i < MAX_AGENTS; i++)
    obs_free(&agents[i].last_obs);

  stop_workers();
  episode_free(&g_rb);
  CloseWindow();
  return 0;
}
