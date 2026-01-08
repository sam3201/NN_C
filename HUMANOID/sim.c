#include "../SAM/SAM.h"
#include "../utils/NN/MUZE/all.h"
#include "../utils/Raylib/src/raylib.h"
#include "../utils/Raylib/src/raymath.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// =======================
// CONFIG
// =======================
#define FPS 60.0

#define EPISODE_SECONDS 8.0f
#define HEAD_TOUCH_EPS 1.5f   // how close to ground counts as “touch”
#define HEAD_TARGET_ALT 90.0f // px above ground that we’d like (tune)

#define OBS_DIM 72

#define WORKER_COUNT 4
#define MAX_AGENTS 64

#define FIXED_DT (1.0f / 120.0f)
#define MAX_ACCUM_DT (0.25f)

// =======================
// JUMP KING ENV
// =======================
#define PLATFORM_MAX 256
#define WORLD_HEIGHT 8000.0f // tall tower
#define GRAVITY_Y 2600.0f
#define MOVE_SPEED 260.0f
#define AIR_CONTROL 0.35f

#define JUMP_CHARGE_RATE 1.6f // per second
#define JUMP_CHARGE_MAX 1.0f
#define JUMP_VY_MIN -650.0f
#define JUMP_VY_MAX -1550.0f
#define JUMP_VX_MAX 520.0f

typedef struct {
  float x, y, w, h; // axis-aligned rect
  int one_way;      // 1 = one-way from below
} Platform;

typedef struct {
  Vector2 pos;
  Vector2 vel;
  float radius;

  int on_ground; // standing on something (ground or platform)
  int was_below; // helper for one-way landing
  float prev_y;  // for one-way crossing test

  // jump charge
  int charging;
  float charge; // 0..1
} PlayerJK;

static Platform g_plats[PLATFORM_MAX];
static int g_plat_count = 0;

enum {
  ACT_NONE = 0,
  ACT_LEFT,
  ACT_RIGHT,
  ACT_CHARGE,  // hold
  ACT_RELEASE, // release jump
  ACTION_COUNT
};

typedef struct {
  float obs[OBS_DIM];
  int n;
} ObsFixed;

typedef struct {
  bool alive;
  Vector2 spawn_origin;

  PlayerJK pl;

  float episode_time;
  float episode_limit;

  // best achieved height this episode (higher = better)
  float best_alt; // altitude above ground baseline

  // --- SAM/MUZE ---
  SAM_t *sam;
  MuCortex *cortex;
  int last_action;

  float accum_dt;
  float control_timer;
  float control_period;

  float pending_reward;
  ObsFixed last_obs;
  int has_last_transition;

  float reward_accumulator;
} Agent;

/* =======================
   GLOBAL CONFIG
======================= */
static int SCREEN_WIDTH = 1280;
static int SCREEN_HEIGHT = 800;

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

static inline float vlen(Vector2 v) { return sqrtf(v.x * v.x + v.y * v.y); }

static inline Vector2 vsub(Vector2 a, Vector2 b) {
  return (Vector2){a.x - b.x, a.y - b.y};
}

static inline Vector2 vadd(Vector2 a, Vector2 b) {
  return (Vector2){a.x + b.x, a.y + b.y};
}

static inline Vector2 vmul(Vector2 a, float s) {
  return (Vector2){a.x * s, a.y * s};
}

static inline float clampf(float x, float a, float b) {
  return x < a ? a : (x > b ? b : x);
}
static inline float saturate(float x) { return clampf(x, -1.0f, 1.0f); }

static int circle_overlaps_rect(Vector2 c, float r, const Platform *pl) {
  float cx = clampf(c.x, pl->x, pl->x + pl->w);
  float cy = clampf(c.y, pl->y, pl->y + pl->h);
  float dx = c.x - cx;
  float dy = c.y - cy;
  return (dx * dx + dy * dy) <= r * r;
}

static void solve_player_platforms(PlayerJK *p) {
  p->on_ground = 0;

  for (int i = 0; i < g_plat_count; i++) {
    Platform *pl = &g_plats[i];

    // one-way: only collide when falling AND we were above the top last step
    if (pl->one_way) {
      if (p->vel.y < 0)
        continue; // rising => pass through

      float top = pl->y;
      float prev_bottom = p->prev_y + p->radius;
      float cur_bottom = p->pos.y + p->radius;

      // must cross the top surface downward
      if (!(prev_bottom <= top && cur_bottom >= top))
        continue;

      // must be horizontally over the platform
      if (p->pos.x < pl->x - p->radius || p->pos.x > pl->x + pl->w + p->radius)
        continue;

      // snap onto top
      p->pos.y = top - p->radius;
      p->vel.y = 0;
      p->on_ground = 1;
      continue;
    }

    // solid platforms (ground)
    if (circle_overlaps_rect(p->pos, p->radius, pl)) {
      // simple: push up out of it (good enough for ground)
      p->pos.y = pl->y - p->radius;
      p->vel.y = 0;
      p->on_ground = 1;
    }
  }
}

static void apply_action_muscles(Agent *a, int action) {
  // Reset to neutral every substep (muscles are per-step, not permanent)
  // We'll start from original rest each time; store "base rest" separately if
  // you want. For now: we do small relative tweaks on a->jt[].rest directly,
  // then restore after solve.
}

// PD controller for joints: tau = kp*(target - angle) - kd*angVel
static inline void obs_pushf(ObsFixed *o, float v) {
  if (o->n < OBS_DIM)
    o->obs[o->n++] = v;
}

static inline float safe_div(float a, float b) {
  return (fabsf(b) > 1e-6f) ? (a / b) : 0.0f;
}

static void build_platforms(void) {
  g_plat_count = 0;

  // ground
  g_plats[g_plat_count++] = (Platform){.x = -2000,
                                       .y = (float)SCREEN_HEIGHT - 40,
                                       .w = 5000,
                                       .h = 40,
                                       .one_way = 0};

  // tower platforms going upward (y decreases as you go up)
  float y = (float)SCREEN_HEIGHT - 120.0f;
  float x = 200.0f;

  for (int i = 0; i < PLATFORM_MAX - 1; i++) {
    float w = 120 + (rand() % 120);
    float h = 14;
    float dx = (rand() % 240) - 120; // random sideways
    float dy = 90 + (rand() % 80);   // step up

    x = clampf(x + dx, 80.0f, (float)SCREEN_WIDTH - 200.0f);
    y -= dy;

    g_plats[g_plat_count++] =
        (Platform){.x = x, .y = y, .w = w, .h = h, .one_way = 1};

    if (y < -WORLD_HEIGHT)
      break;
  }
}

static void encode_observation_jk(const Agent *a, ObsFixed *out) {
  out->n = 0;

  const PlayerJK *p = &a->pl;

  // normalize
  float px = p->pos.x / (float)SCREEN_WIDTH;
  float py = p->pos.y / (float)SCREEN_HEIGHT; // note: tower goes negative
  float vx = p->vel.x / 800.0f;
  float vy = p->vel.y / 1800.0f;

  obs_pushf(out, px);
  obs_pushf(out, py);
  obs_pushf(out, vx);
  obs_pushf(out, vy);
  obs_pushf(out, (float)p->on_ground);
  obs_pushf(out, p->charge);

  // nearest platforms above: (dx, dy, w)
  // dy is negative when above since y smaller => platform above => (plat.y -
  // p.y) < 0
  int found = 0;
  for (int i = 0; i < g_plat_count && found < 3; i++) {
    const Platform *pl = &g_plats[i];
    if (!pl->one_way)
      continue;
    if (pl->y >= p->pos.y)
      continue; // only above

    float dx = (pl->x + pl->w * 0.5f) - p->pos.x;
    float dy = pl->y - p->pos.y;

    // take "closest above" by dy magnitude
    // simple pass: just record first few; better: scan for smallest |dy|
    obs_pushf(out, dx / (float)SCREEN_WIDTH);
    obs_pushf(out, dy / 600.0f);
    obs_pushf(out, pl->w / 200.0f);
    found++;
  }

  while (found < 3) {
    obs_pushf(out, 0);
    obs_pushf(out, 0);
    obs_pushf(out, 0);
    found++;
  }

  obs_pushf(out, 1.0f); // bias

  for (int k = out->n; k < OBS_DIM; k++)
    out->obs[k] = 0.0f;
  out->n = OBS_DIM;
}

static void init_agent(Agent *a, Vector2 origin) {
  memset(a, 0, sizeof(*a));
  a->alive = true;
  a->pl.pos = origin;
  a->pl.vel = (Vector2){0, 0};
  a->pl.radius = 10.0f;
  a->pl.on_ground = 1;
  a->pl.charging = 0;
  a->pl.charge = 0.0f;
  a->pl.prev_y = a->pl.pos.y;
  a->best_alt = 0.0f;
}

void init_agents(void) {
  MuConfig cfg = {.obs_dim = OBS_DIM,
                  .latent_dim = 64, // not used by SAM_init, ok to keep here
                  .action_count = ACTION_COUNT};

  int cols = 16;
  float spacing = 70.0f;

  for (int i = 0; i < MAX_AGENTS; i++) {
    Agent *a = &agents[i];
    memset(a, 0, sizeof(*a));

    a->alive = true;

    // Fixed-step / control loop
    a->accum_dt = 0.0f;
    a->control_period = 1.0f / 30.0f; // 30 Hz decisions
    a->control_timer = 0.0f;
    a->last_action = ACT_NONE;
    a->reward_accumulator = 0.0f;
    a->pending_reward = 0.0f;
    a->has_last_transition = 0;
    memset(&a->last_obs, 0, sizeof(a->last_obs));

    // Spawn pose/location
    int cx = i % cols;
    int cy = i / cols;
    Vector2 origin = {SCREEN_WIDTH * 0.2f + cx * spacing,
                      SCREEN_HEIGHT * 0.35f + cy * spacing * 1.1f};

    a->spawn_origin = origin;

    a->episode_time = 0.0f;
    a->episode_limit = EPISODE_SECONDS;

    // Brain
    a->sam = SAM_init(cfg.obs_dim, cfg.action_count, 4, 0);
    a->cortex = SAM_as_MUZE(a->sam);

    // Optional: tweak MUZE sampler behavior per agent
    if (a->cortex) {
      a->cortex->policy_epsilon = 0.10f;
      a->cortex->policy_temperature = 1.0f;
      a->cortex->use_mcts = false;
    }
  }
}

static void reset_agent_episode(Agent *a) {
  if (a->cortex && a->has_last_transition) {
    a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                     a->last_action, a->pending_reward, 1);
  }

  a->alive = true;
  a->episode_time = 0.0f;
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

  a->best_alt = 0.0f;
}

static void solve_joint(Agent *ag, const Joint *j) {
  Particle *A = &ag->pt[j->a];
  Particle *B = &ag->pt[j->b];

  Vector2 delta = vsub(B->p, A->p);
  float d = vlen(delta);
  if (d < 1e-6f)
    return;

  float diff = (d - j->rest) / d; // normalized stretch

  // Weighted by mass
  float wA = A->invMass;
  float wB = B->invMass;
  float wsum = wA + wB;
  if (wsum <= 0.0f)
    return;

  // Stiffness factor (1.0 = fully enforce in one shot)
  float k = ag->joint_stiffness;

  Vector2 corr = vmul(delta, k * diff);

  // Move each end opposite directions
  if (wA > 0.0f)
    A->p = vadd(A->p, vmul(corr, (wA / wsum)));
  if (wB > 0.0f)
    B->p = vsub(B->p, vmul(corr, (wB / wsum)));
}

static void ground_collide(Particle *p, float groundY) {
  if (p->invMass <= 0.0f)
    return;
  if (p->p.y > groundY) {
    p->p.y = groundY;

    // crude friction/damping: reduce horizontal velocity after collision
    Vector2 v = vsub(p->p, p->pprev);
    v.x *= 0.6f;
    v.y *= 0.0f; // kill vertical motion on ground
    p->pprev = vsub(p->p, v);
  }
}

void update_agent(Agent *a) {
  float dt = clampf(g_dt, 0.0f, MAX_ACCUM_DT);
  a->accum_dt += dt;
  a->episode_time += dt;

  // CONTROL TICK
  a->control_timer -= dt;
  if (a->control_timer <= 0.0f) {
    a->control_timer +=
        a->control_period > 0 ? a->control_period : (1.0f / 20.0f);

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

    // movement intent
    float ax = 0.0f;
    if (a->last_action == ACT_LEFT)
      ax = -MOVE_SPEED;
    if (a->last_action == ACT_RIGHT)
      ax = +MOVE_SPEED;

    float control = p->on_ground ? 1.0f : AIR_CONTROL;
    p->vel.x += ax * control * FIXED_DT;

    // simple drag
    p->vel.x *= p->on_ground ? 0.92f : 0.985f;

    // charging
    if (a->last_action == ACT_CHARGE && p->on_ground) {
      p->charging = 1;
      p->charge = clampf(p->charge + JUMP_CHARGE_RATE * FIXED_DT, 0.0f,
                         JUMP_CHARGE_MAX);
    }

    // release jump
    if (a->last_action == ACT_RELEASE && p->on_ground) {
      float t = clampf(p->charge, 0.0f, 1.0f);
      float vy = JUMP_VY_MIN + (JUMP_VY_MAX - JUMP_VY_MIN) * t; // negative (up)
      float vx = clampf(p->vel.x, -JUMP_VX_MAX, JUMP_VX_MAX);

      p->vel.y = vy;
      p->vel.x = vx;

      p->on_ground = 0;
      p->charging = 0;
      p->charge = 0.0f;
    }

    // gravity
    p->vel.y += GRAVITY_Y * FIXED_DT;

    // integrate
    p->pos.x += p->vel.x * FIXED_DT;
    p->pos.y += p->vel.y * FIXED_DT;

    // bounds
    if (p->pos.x < 0) {
      p->pos.x = 0;
      p->vel.x = 0;
    }
    if (p->pos.x > SCREEN_WIDTH) {
      p->pos.x = SCREEN_WIDTH;
      p->vel.x = 0;
    }

    // collide platforms (one-way landing)
    solve_player_platforms(p);

    // REWARD: maximize height (lower y is higher). Pick a baseline: screen
    // ground.
    float groundY = (float)SCREEN_HEIGHT - 40.0f;
    float alt = groundY - p->pos.y; // higher alt => better
    if (alt > a->best_alt) {
      float delta = alt - a->best_alt;
      a->best_alt = alt;
      a->pending_reward += 0.02f * (delta / 50.0f); // shaped for progress
    }

    // small alive reward
    a->pending_reward += 0.001f;

    // punish falling too low (death)
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

void update_agent(Agent *a) {
  if (!a || !a->alive)
    return;

  float dt = g_dt;
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f;
  dt = clampf(dt, 0.0f, MAX_ACCUM_DT);
  a->accum_dt += dt;
  a->episode_time += dt;

  const float groundY = (float)SCREEN_HEIGHT - 40.0f;
  const Vector2 gravity = (Vector2){0.0f, 1400.0f}; // y-down in raylib

  // ------------------------------------------------------------
  // 1) CONTROL TICK (30Hz): learn from previous action, then choose new
  // ------------------------------------------------------------
  a->control_timer -= dt;
  if (a->control_timer <= 0.0f) {
    a->control_timer +=
        (a->control_period > 0.0f) ? a->control_period : (1.0f / 30.0f);

    // (A) If we have a previous (obs, action), train it with reward accumulated
    // since then.
    if (a->cortex && a->has_last_transition) {
      int terminal =
          a->alive ? 0 : 1; // usually 0 here; final flush happens below too
      a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                       a->last_action, a->pending_reward, terminal);
    }

    // reset reward accumulator for the *next* action window
    a->pending_reward = 0.0f;

    // (B) Build current obs and pick a new action
    encode_observation_humanoid(a, groundY, &a->last_obs);

    if (a->cortex) {
      a->last_action = muze_plan(a->cortex, a->last_obs.obs, (size_t)OBS_DIM,
                                 (size_t)ACTION_COUNT);
    } else {
      a->last_action = ACT_NONE;
    }

    a->has_last_transition = 1;
  }

  // ------------------------------------------------------------
  // 2) PHYSICS SUBSTEPS (120Hz fixed dt): integrate + muscles + constraints
  // ------------------------------------------------------------
  while (a->accum_dt >= FIXED_DT) {
    a->accum_dt -= FIXED_DT;

    // --- Verlet integrate (gravity) ---
    for (int i = 0; i < P_COUNT; i++) {
      Particle *p = &a->pt[i];
      if (p->invMass <= 0.0f)
        continue;

      Vector2 cur = p->p;
      Vector2 v = vsub(p->p, p->pprev);

      v = vmul(v, 0.995f); // damping

      Vector2 next = vadd(vadd(cur, v), vmul(gravity, FIXED_DT * FIXED_DT));
      p->pprev = cur;
      p->p = next;
    }

    // --- Apply action as "muscles" ---
    float rest0[J_COUNT];
    for (int j = 0; j < J_COUNT; j++)
      rest0[j] = a->jt[j].rest;

    float contract = 0.92f;
    float extend = 1.08f;

    switch (a->last_action) {
    case ACT_STEP_LEFT: {
      a->jt[J_HIP_L].rest *= contract;
      a->jt[J_SHIN_L].rest *= contract;
      a->jt[J_HIP_R].rest *= extend;
      a->jt[J_SHIN_R].rest *= extend;
      verlet_impulse(&a->pt[P_HIP], (Vector2){-18.0f * FIXED_DT, 0.0f});
    } break;

    case ACT_STEP_RIGHT: {
      a->jt[J_HIP_R].rest *= contract;
      a->jt[J_SHIN_R].rest *= contract;
      a->jt[J_HIP_L].rest *= extend;
      a->jt[J_SHIN_L].rest *= extend;
      verlet_impulse(&a->pt[P_HIP], (Vector2){+18.0f * FIXED_DT, 0.0f});
    } break;

    case ACT_KNEE_L_UP: {
      a->jt[J_SHIN_L].rest *= contract;
      verlet_impulse(&a->pt[P_ANKLE_L], (Vector2){0.0f, -55.0f * FIXED_DT});
    } break;

    case ACT_KNEE_R_UP: {
      a->jt[J_SHIN_R].rest *= contract;
      verlet_impulse(&a->pt[P_ANKLE_R], (Vector2){0.0f, -55.0f * FIXED_DT});
    } break;

    case ACT_ARMS_UP: {
      a->jt[J_UPPERARM_L].rest *= contract;
      a->jt[J_FOREARM_L].rest *= contract;
      a->jt[J_UPPERARM_R].rest *= contract;
      a->jt[J_FOREARM_R].rest *= contract;
      verlet_impulse(&a->pt[P_HAND_L], (Vector2){0.0f, -45.0f * FIXED_DT});
      verlet_impulse(&a->pt[P_HAND_R], (Vector2){0.0f, -45.0f * FIXED_DT});
    } break;

    case ACT_ARMS_DOWN: {
      a->jt[J_UPPERARM_L].rest *= extend;
      a->jt[J_FOREARM_L].rest *= extend;
      a->jt[J_UPPERARM_R].rest *= extend;
      a->jt[J_FOREARM_R].rest *= extend;
    } break;

    case ACT_SQUAT: {
      a->jt[J_SPINE1].rest *= contract;
      a->jt[J_SPINE2].rest *= contract;
      a->jt[J_HIP_L].rest *= contract;
      a->jt[J_SHIN_L].rest *= contract;
      a->jt[J_HIP_R].rest *= contract;
      a->jt[J_SHIN_R].rest *= contract;
    } break;

    case ACT_STAND: {
      a->jt[J_SPINE1].rest *= extend;
      a->jt[J_SPINE2].rest *= extend;
      a->jt[J_HIP_L].rest *= extend;
      a->jt[J_SHIN_L].rest *= extend;
      a->jt[J_HIP_R].rest *= extend;
      a->jt[J_SHIN_R].rest *= extend;
    } break;

    default:
      break;
    }

    // --- Solve constraints ---
    int iters = 10;
    float old_stiff = a->joint_stiffness;
    a->joint_stiffness = 1.0f;

    for (int it = 0; it < iters; it++) {
      for (int j = 0; j < J_COUNT; j++)
        solve_joint(a, &a->jt[j]);
      for (int i = 0; i < P_COUNT; i++)
        ground_collide(&a->pt[i], groundY);
    }

    a->joint_stiffness = old_stiff;

    for (int j = 0; j < J_COUNT; j++)
      a->jt[j].rest = rest0[j];

    // --- Reward (per substep, accumulated to control tick) ---
    float reward = 0.0f;

    float head_y = a->pt[P_HEAD].p.y;
    float hip_y = a->pt[P_HIP].p.y;

    // alive drip
    reward += 0.001f;

    // HEAD ALTITUDE (y-down, so altitude = groundY - head_y)
    float head_alt = groundY - head_y; // px above ground
    float alt_norm = clampf(head_alt / HEAD_TARGET_ALT, 0.0f, 1.0f);
    reward += 0.004f * alt_norm;

    // penalize “collapsed” hip a bit (optional)
    if (hip_y > groundY - 25.0f)
      reward -= 0.01f;

    // BIG punishment if head touches ground (and we’ll terminate)
    if (head_y >= groundY - HEAD_TOUCH_EPS) {
      reward -= 0.50f;  // strong negative signal
      a->alive = false; // terminal this step
    }

    // accumulate reward until next control tick
    a->pending_reward += reward;
    a->reward_accumulator += reward;
  }

  // ------------------------------------------------------------
  // 3) TERMINATION + FINAL LEARN FLUSH
  // ------------------------------------------------------------
  // Episode timeout
  if (a->episode_time >= a->episode_limit) {
    // treat timeout as terminal (or non-terminal; terminal usually works better
    // for episodic training)
    reset_agent_episode(a, groundY);
    return;
  }

  // If died (head touch), reset immediately
  if (!a->alive) {
    reset_agent_episode(a, groundY);
    return;
  }
}

static void draw_platforms(float camY) {
  for (int i = 0; i < g_plat_count; i++) {
    Platform *pl = &g_plats[i];
    DrawRectangle((int)pl->x, (int)(pl->y + camY), (int)pl->w, (int)pl->h,
                  pl->one_way ? GRAY : DARKGRAY);
  }
}

static void draw_agent_jk(const Agent *a, float camY) {
  Vector2 p = a->pl.pos;
  DrawCircle((int)p.x, (int)(p.y + camY), (int)a->pl.radius, RAYWHITE);
}

pthread_cond_broadcast(&job_cv);

while (job_active) {
  pthread_cond_wait(&done_cv, &job_mtx);
}
pthread_mutex_unlock(&job_mtx);
}

static void *agent_worker(void *arg) {
  (void)arg;

  for (;;) {
    // Wait for a job batch to become active (or quit)
    pthread_mutex_lock(&job_mtx);
    while (!job_active && !job_quit) {
      pthread_cond_wait(&job_cv, &job_mtx);
    }
    if (job_quit) {
      pthread_mutex_unlock(&job_mtx);
      break;
    }
    pthread_mutex_unlock(&job_mtx);

    // Work loop: grab next agent index atomically under mutex
    for (;;) {
      int idx;

      pthread_mutex_lock(&job_mtx);
      idx = job_next_agent++;
      pthread_mutex_unlock(&job_mtx);

      if (idx >= MAX_AGENTS)
        break;

      update_agent(&agents[idx]);
    }

    // Signal completion for this worker
    pthread_mutex_lock(&job_mtx);
    job_done_workers++;

    if (job_done_workers >= WORKER_COUNT) {
      job_active = 0;                // batch finished
      pthread_cond_signal(&done_cv); // wake main thread
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

/* =======================
   MAIN
======================= */
int main(void) {
  srand(time(NULL));

  InitWindow(1280, 800, "HUMANOID");
  // SetExitKey(KEY_NULL); //
  SCREEN_WIDTH = GetScreenWidth();
  SCREEN_HEIGHT = GetScreenHeight();
  SetTargetFPS(60);

  build_platforms();

  init_agents();

  start_workers();

  /*
  for (int y = 0; y < WORLD_SIZE; y++) {
    pthread_rwlock_init(&world[x][y].lock, NULL);
    world[x][y].generated = false;
    world[x][y].resource_count = 0;
    world[x][y].mob_spawn_timer = 0.0f;
  }
  */

  while (!WindowShouldClose()) {
    g_dt = GetFrameTime();

    float groundY = (float)SCREEN_HEIGHT - 40.0f;
    float camY = 0.0f;
    float focus_alt = agents[0].best_alt; // or max across agents
    camY = clampf((groundY - agents[0].pl.pos.y) - 250.0f, 0.0f, WORLD_HEIGHT);
    camY = -camY; // move world down as you go up

    run_agent_jobs();

    BeginDrawing();
    ClearBackground(BLACK);

    draw_platforms(camY);
    for (int i = 0; i < MAX_AGENTS; i++)
      draw_agent_jk(&agents[i], camY);

    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      draw_agent(&agents[i]);
    }

    DrawText("HUMANOID Simulation", 20, 160, 20, RAYWHITE);
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 185, 20, RAYWHITE);

    EndDrawing();
  }

  stop_workers();

  CloseWindow();
  return 0;
}
