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

#define OBS_DIM 71

#define WORKER_COUNT 4
#define MAX_AGENTS 128

#define FIXED_DT                                                               \
  (1.0f / 120.0f)            // stable for humanoids; 60 can work but is touchy
#define MAX_ACCUM_DT (0.25f) // prevent spiral-of-death
#define MAX_TORQUE (120.0f)  // clamp to avoid explosions
#define JOINT_DAMPING (1.2f) // extra damping for stability
#define ANGVEL_CLAMP (40.0f) // clamp angular velocity (optional safety)

// =======================
// SIMPLE VERLET RAGDOLL
// =======================

typedef struct {
  Vector2 p;     // current position
  Vector2 pprev; // previous position (for Verlet)
  float invMass; // 0 = pinned, 1 = normal
} Particle;

typedef struct {
  int a;      // particle index A
  int b;      // particle index B
  float rest; // rest length
} Joint;

typedef struct {
  float obs[OBS_DIM];
  int n;
} ObsFixed;

enum {
  P_HEAD = 0,
  P_NECK,
  P_CHEST,
  P_HIP,

  P_SHOULDER_L,
  P_ELBOW_L,
  P_HAND_L,

  P_SHOULDER_R,
  P_ELBOW_R,
  P_HAND_R,

  P_KNEE_L,
  P_ANKLE_L,

  P_KNEE_R,
  P_ANKLE_R,

  P_COUNT
};

enum {
  J_NECK = 0,
  J_SPINE1,
  J_SPINE2,

  J_SHOULDER_L,
  J_UPPERARM_L,
  J_FOREARM_L,

  J_SHOULDER_R,
  J_UPPERARM_R,
  J_FOREARM_R,

  J_HIP_L,
  J_SHIN_L,

  J_HIP_R,
  J_SHIN_R,

  J_COUNT
};

typedef struct {
  bool alive;

  Particle pt[P_COUNT];
  Joint jt[J_COUNT];

  float joint_stiffness; // 0..1

  // --- SAM / MUZE control ---
  SAM_t *sam;
  MuCortex *cortex;
  int last_action;

  // --- fixed step / control rate ---
  float accum_dt;
  float control_timer;
  float control_period; // e.g. 1/30

  // --- learning bookkeeping (control-rate) ---
  float pending_reward;    // accumulated reward since last decision
  ObsFixed last_obs;       // observation used to pick last_action
  int has_last_transition; // whether last_obs/last_action is valid

  float reward_accumulator;

} Agent;

enum {
  ACT_NONE = 0,

  ACT_STEP_LEFT,
  ACT_STEP_RIGHT,

  ACT_KNEE_L_UP,
  ACT_KNEE_R_UP,

  ACT_ARMS_UP,
  ACT_ARMS_DOWN,

  ACT_SQUAT,
  ACT_STAND,

  ACTION_COUNT
};

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

static inline void verlet_impulse(Particle *p, Vector2 dv) {
  // In Verlet, changing pprev changes velocity.
  // v = p - pprev, so to add dv: set pprev = p - (v + dv) = pprev - dv
  p->pprev = vsub(p->pprev, dv);
}

static void apply_action_muscles(Agent *a, int action) {
  // Reset to neutral every substep (muscles are per-step, not permanent)
  // We'll start from original rest each time; store "base rest" separately if
  // you want. For now: we do small relative tweaks on a->jt[].rest directly,
  // then restore after solve.
}

// PD controller for joints: tau = kp*(target - angle) - kd*angVel
static inline float joint_pd(float angle, float angVel, float target, float kp,
                             float kd) {
  float err = target - angle;
  return kp * err - kd * angVel;
}

static inline void apply_joint_torque(Joint *j, float tau) {
  tau = clampf(tau, -MAX_TORQUE, MAX_TORQUE);
  // Your engine call here:
  // physics_apply_joint_torque(j, tau);
  j->motor_torque = tau; // placeholder
}

static inline void damp_body(Body *b, float dt) {
  // optional global damping (helps ragdolls)
  b->linVel.x *= 1.0f / (1.0f + 0.05f * dt);
  b->linVel.y *= 1.0f / (1.0f + 0.05f * dt);
  b->angVel *= 1.0f / (1.0f + JOINT_DAMPING * dt);

  // optional clamps (safety)
  b->angVel = clampf(b->angVel, -ANGVEL_CLAMP, ANGVEL_CLAMP);
}

static inline void obs_pushf(ObsFixed *o, float v) {
  if (o->n < OBS_DIM)
    o->obs[o->n++] = v;
}

static inline float safe_div(float a, float b) {
  return (fabsf(b) > 1e-6f) ? (a / b) : 0.0f;
}

static void make_joint(Agent *ag, int j, int a, int b) {
  ag->jt[j].a = a;
  ag->jt[j].b = b;
  ag->jt[j].rest = vlen(vsub(ag->pt[b].p, ag->pt[a].p));
}

static void encode_observation_humanoid(const Agent *a, float groundY,
                                        ObsFixed *out) {
  memset(out, 0, sizeof(*out));
  out->n = 0;

  Vector2 hip = a->pt[P_HIP].p;

  // For each particle: (pos relative to hip) + (vel)
  // Normalize a bit so values aren't huge.
  const float pos_scale = 1.0f / 100.0f;
  const float vel_scale = 1.0f / 50.0f;

  for (int i = 0; i < P_COUNT; i++) {
    Vector2 p = a->pt[i].p;
    Vector2 v = vsub(a->pt[i].p, a->pt[i].pprev); // verlet velocity proxy

    obs_pushf(out, (p.x - hip.x) * pos_scale);
    obs_pushf(out, (p.y - hip.y) * pos_scale);
    obs_pushf(out, v.x * vel_scale);
    obs_pushf(out, v.y * vel_scale);

    // contact bit (foot/ankle touching ground)
    float contact = 0.0f;
    if (i == P_ANKLE_L || i == P_ANKLE_R) {
      contact = (p.y >= groundY - 0.5f) ? 1.0f : 0.0f;
    }
    obs_pushf(out, contact);
  }

  // A small bias
  obs_pushf(out, 1.0f);

  // Remaining slots already zero-filled
}

static void init_agent(Agent *a, Vector2 origin) {
  memset(a, 0, sizeof(*a));
  a->alive = true;
  a->joint_stiffness = 1.0f;

  // A simple standing-ish T pose (y increases downward in Raylib)
  // Feel free to tweak these numbers.
  a->pt[P_HEAD].p = vadd(origin, (Vector2){0, -90});
  a->pt[P_NECK].p = vadd(origin, (Vector2){0, -75});
  a->pt[P_CHEST].p = vadd(origin, (Vector2){0, -50});
  a->pt[P_HIP].p = vadd(origin, (Vector2){0, -20});

  a->pt[P_SHOULDER_L].p = vadd(origin, (Vector2){-25, -65});
  a->pt[P_ELBOW_L].p = vadd(origin, (Vector2){-50, -55});
  a->pt[P_HAND_L].p = vadd(origin, (Vector2){-70, -45});

  a->pt[P_SHOULDER_R].p = vadd(origin, (Vector2){25, -65});
  a->pt[P_ELBOW_R].p = vadd(origin, (Vector2){50, -55});
  a->pt[P_HAND_R].p = vadd(origin, (Vector2){70, -45});

  a->pt[P_KNEE_L].p = vadd(origin, (Vector2){-15, 15});
  a->pt[P_ANKLE_L].p = vadd(origin, (Vector2){-15, 55});

  a->pt[P_KNEE_R].p = vadd(origin, (Vector2){15, 15});
  a->pt[P_ANKLE_R].p = vadd(origin, (Vector2){15, 55});

  // Verlet needs previous position initialized
  for (int i = 0; i < P_COUNT; i++) {
    a->pt[i].pprev = a->pt[i].p;
    a->pt[i].invMass = 1.0f;
  }

  // If you want to "pin" something for debugging:
  // a->pt[P_HEAD].invMass = 0.0f;

  // Bones / constraints
  make_joint(a, J_NECK, P_HEAD, P_NECK);
  make_joint(a, J_SPINE1, P_NECK, P_CHEST);
  make_joint(a, J_SPINE2, P_CHEST, P_HIP);

  make_joint(a, J_SHOULDER_L, P_NECK, P_SHOULDER_L);
  make_joint(a, J_UPPERARM_L, P_SHOULDER_L, P_ELBOW_L);
  make_joint(a, J_FOREARM_L, P_ELBOW_L, P_HAND_L);

  make_joint(a, J_SHOULDER_R, P_NECK, P_SHOULDER_R);
  make_joint(a, J_UPPERARM_R, P_SHOULDER_R, P_ELBOW_R);
  make_joint(a, J_FOREARM_R, P_ELBOW_R, P_HAND_R);

  make_joint(a, J_HIP_L, P_HIP, P_KNEE_L);
  make_joint(a, J_SHIN_L, P_KNEE_L, P_ANKLE_L);

  make_joint(a, J_HIP_R, P_HIP, P_KNEE_R);
  make_joint(a, J_SHIN_R, P_KNEE_R, P_ANKLE_R);
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
    a->joint_stiffness = 1.0f;

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

    init_agent(a, origin); // <-- your particle/joint setup

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
  if (!a || !a->alive)
    return;

  float dt = g_dt;
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f;
  dt = clampf(dt, 0.0f, MAX_ACCUM_DT);
  a->accum_dt += dt;

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
    a->last_obs.n = OBS_DIM; // ensure dimension is consistent

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

    // --- Reward (per substep, but ACCUMULATED) ---
    float reward = 0.0f;
    float hip_y = a->pt[P_HIP].p.y;
    float head_y = a->pt[P_HEAD].p.y;

    reward += 0.001f; // alive bonus
    if (head_y >= groundY - 2.0f)
      reward -= 0.05f;
    if (hip_y > groundY - 25.0f)
      reward -= 0.01f;

    // accumulate reward until next control tick
    a->pending_reward += reward;
    a->reward_accumulator += reward;
  }

  // ------------------------------------------------------------
  // 3) TERMINATION + FINAL LEARN FLUSH
  // ------------------------------------------------------------
  if (a->pt[P_HEAD].p.y >= ((float)SCREEN_HEIGHT - 42.0f)) {
    // Mark dead
    a->alive = false;

    // One last learn call to close the episode
    if (a->cortex && a->has_last_transition) {
      a->cortex->learn(a->cortex->brain, a->last_obs.obs, (size_t)OBS_DIM,
                       a->last_action, a->pending_reward, 1 /*terminal*/);
    }
    a->pending_reward = 0.0f;
    a->has_last_transition = 0;
  }
}

static void draw_agent(const Agent *a) {
  if (!a || !a->alive)
    return;

  // bones
  for (int j = 0; j < J_COUNT; j++) {
    const Joint *jt = &a->jt[j];
    Vector2 A = a->pt[jt->a].p;
    Vector2 B = a->pt[jt->b].p;
    DrawLineV(A, B, RAYWHITE);
  }

  // joints as circles
  for (int i = 0; i < P_COUNT; i++) {
    float r = 4.0f;
    if (i == P_HEAD)
      r = 8.0f;
    DrawCircleV(a->pt[i].p, r, RAYWHITE);
  }
}

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

    run_agent_jobs();

    BeginDrawing();
    ClearBackground(BLACK);

    float groundY = (float)SCREEN_HEIGHT - 40.0f;
    DrawLine(0, (int)groundY, SCREEN_WIDTH, (int)groundY, DARKGRAY);

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
