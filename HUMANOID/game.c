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
#define WORKER_COUNT 4
#define MAX_AGENTS 128

typedef struct {
  int x;
  int y;
  struct joint *connection;
} joint;

typedef struct {
  bool alive;
  int x;
  int y;

  // Head
  struct {
    int x;
    int y;
  } Head;

  // Neck
  struct {
    int x;
    int y;
  } Neck;

  // Body
  struct {
    int x;
    int y;
  } Body;

  // Legs
  struct {
    int x;
    int y;
  } Legs;

  // Arms
  struct {
    int x;
    int y;
  } Arms;

  // Feet
  struct {
    int x;
    int y;
  } Feet;

  // Hands
  struct {
    int x;
    int y;
  } Hands;

  // Joints
  struct joint *(struct)Head;
  struct joint *Neck;
  struct joint *Body;
  struct joint *Legs;
  struct joint *Arms;
  struct joint *Feet;
  struct joint *Hands;

} Agent;

/* =======================
   GLOBAL CONFIG
======================= */
const unsigned int SCREEN_WIDTH = 1280;
const unsigned int SCREEN_HEIGHT = 800;

pthread_mutex_t job_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t job_cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t done_cv = PTHREAD_COND_INITIALIZER;
bool job_quit = 0;
bool job_active = 0;
unsigned int job_next_agent = 0;
size_t job_done_workers = 0;
pthread_t workers[WORKER_COUNT];

Agent agents[MAX_AGENTS];

static void init_agents(void) {
  for (int i = 0; i < MAX_AGENTS; i++) {
    agents[i].alive = true;
    agents[i].x = SCREEN_WIDTH / 2;
    agents[i].y = SCREEN_HEIGHT / 2;
    agents[i].Head.x = agents[i].x;
    agents[i].Head.y = agents[i].y - 20;
    agents[i].Neck.x = agents[i].x;
    agents[i].Neck.y = agents[i].y - 40;
    agents[i].Body.x = agents[i].x;
    agents[i].Body.y = agents[i].y - 60;
    agents[i].Legs.x = agents[i].x;
    agents[i].Legs.y = agents[i].y - 80;
    agents[i].Arms.x = agents[i].x;
    agents[i].Arms.y = agents[i].y - 100;
    agents[i].Feet.x = agents[i].x;
    agents[i].Feet.y = agents[i].y - 120;
    agents[i].Hands.x = agents[i].x;
    agents[i].Hands.y = agents[i].y - 140;
  }
}

static void draw_agent(Agent *a, Texture2D *ap, Texture2D *tc) {
  // Head
  DrawCircleV((Vector2){a->x, a->y}, 10, RAYWHITE);

  // Neck
  DrawLine(a->x, a->y, a->x, a->y - 20, RAYWHITE);

  // Body
  DrawLine(a->x, a->y - 20, a->x, a->y - 40, RAYWHITE);

  // Legs
  DrawLine(a->x, a->y - 40, a->x - 10, a->y - 60, RAYWHITE);
  DrawLine(a->x, a->y - 40, a->x + 10, a->y - 60, RAYWHITE);

  // Arms
  DrawLine(a->x, a->y - 20, a->x - 10, a->y - 40, RAYWHITE);
  DrawLine(a->x, a->y - 20, a->x + 10, a->y - 40, RAYWHITE);

  // Feet
  DrawLine(a->x - 10, a->y - 60, a->x - 20, a->y - 80, RAYWHITE);
  DrawLine(a->x + 10, a->y - 60, a->x + 20, a->y - 80, RAYWHITE);

  // Hands
  DrawLine(a->x - 10, a->y - 40, a->x - 20, a->y - 60, RAYWHITE);
  DrawLine(a->x + 10, a->y - 40, a->x + 20, a->y - 60, RAYWHITE);
}

static void update_agent(Agent *a) {}

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

      // update_agent(&agents[idx]);
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
