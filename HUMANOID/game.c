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
#define FPS 60
#define WORKER_COUNT 4
#define MAX_AGENTS 128

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

  // optional per-agent tuning
  float joint_stiffness; // 0..1 (we’ll use ~1)
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

static void make_joint(Agent *ag, int j, int a, int b) {
  ag->jt[j].a = a;
  ag->jt[j].b = b;
  ag->jt[j].rest = vlen(vsub(ag->pt[b].p, ag->pt[a].p));
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

static void init_agents(void) {
  // Scatter them a bit so you can see multiple
  int cols = 16;
  float spacing = 70.0f;

  for (int i = 0; i < MAX_AGENTS; i++) {
    int cx = i % cols;
    int cy = i / cols;
    Vector2 o = {SCREEN_WIDTH * 0.2f + cx * spacing,
                 SCREEN_HEIGHT * 0.35f + cy * spacing * 1.1f};
    init_agent(&agents[i], o);
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

static void update_agent(Agent *a) {
  if (!a || !a->alive)
    return;

  float dt = g_dt;
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f;

  const Vector2 gravity = {0.0f, 1400.0f}; // pixels/sec^2
  const float groundY = (float)SCREEN_HEIGHT - 40.0f;

  // 1) Integrate (Verlet)
  for (int i = 0; i < P_COUNT; i++) {
    Particle *p = &a->pt[i];
    if (p->invMass <= 0.0f)
      continue;

    Vector2 vel = vsub(p->p, p->pprev);
    p->pprev = p->p;

    // a = gravity
    Vector2 acc = gravity;

    // p = p + v + a*dt^2
    p->p = vadd(p->p, vadd(vel, vmul(acc, dt * dt)));
  }

  // 2) Constraint solve (iterate for stability)
  int iterations = 10;
  for (int it = 0; it < iterations; it++) {
    for (int j = 0; j < J_COUNT; j++) {
      solve_joint(a, &a->jt[j]);
    }
    // ground each iteration helps stop “sinking”
    for (int i = 0; i < P_COUNT; i++) {
      ground_collide(&a->pt[i], groundY);
    }
  }

  // 3) Final ground pass
  for (int i = 0; i < P_COUNT; i++) {
    ground_collide(&a->pt[i], groundY);
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
    float dt = GetFrameTime();

    run_agent_jobs();

    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
    }

    BeginDrawing();
    ClearBackground(BLACK);

    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      draw_agent(&agents[i], ap, tc);
    }

    DrawText("HUMANOID Simulation", 20, 160, 20, RAYWHITE);
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 185, 20, RAYWHITE);

    EndDrawing();
  }

  stop_workers();

  CloseWindow();
  return 0;
}
