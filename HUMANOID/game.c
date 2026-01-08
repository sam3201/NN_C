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

#define WORKER_COUNT 4
#define MAX_AGENTS 128

/* =======================
   GLOBAL CONFIG
======================= */

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
  TILE_SIZE = SCREEN_HEIGHT / 18.0f;
  SetTargetFPS(60);

  init_agents();

  start_workers();

  for (int x = 0; x < WORLD_SIZE; x++) {
    for (int y = 0; y < WORLD_SIZE; y++) {
      pthread_rwlock_init(&world[x][y].lock, NULL);
      world[x][y].generated = false;
      world[x][y].resource_count = 0;
      world[x][y].mob_spawn_timer = 0.0f;
    }
  }

  for (int i = 0; i < MAX_PROJECTILES; i++)
    projectiles[i].alive = false;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();

    camera_pos.x += (player.position.x - camera_pos.x) * 0.1f;
    camera_pos.y += (player.position.y - camera_pos.y) * 0.1f;

    if (cam_shake > 0.0f) {
      cam_shake -= dt;
      float mag = cam_shake * 0.65f;
      camera_pos.x += randf(-mag, mag);
      camera_pos.y += randf(-mag, mag);
    }
    WORLD_SCALE = lerp(WORLD_SCALE, target_world_scale, 0.12f);

    update_player();
    update_visible_world(dt);
    update_projectiles(dt);
    update_daynight(dt);
    collect_nearby_pickups();

    update_visible_world(dt);

    g_dt = dt;
    run_agent_jobs();

    // detect transition night->day for reward
    int now_night = is_night_cached;
    if (was_night && !now_night) {
      // dawn reward: shards + small base repair
      inv_shards += 5;
      for (int t = 0; t < TRIBE_COUNT; t++) {
        tribes[t].integrity = fminf(100.0f, tribes[t].integrity + 15.0f);
      }
    }
    was_night = now_night;

    // raid spawner
    if (is_night_cached) {
      raid_timer -= dt;
      if (raid_timer <= 0.0f) {
        raid_timer = raid_interval;
        spawn_raid_wave();
      }
    } else {
      raid_timer = 1.5f;
    }

    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      int acx = (int)(agents[i].position.x / CHUNK_SIZE);
      int acy = (int)(agents[i].position.y / CHUNK_SIZE);
      (void)get_chunk(acx, acy);
    }

    update_pickups(dt);

    BeginDrawing();
    ClearBackground(BLACK);

    if (g_state == STATE_PLAYING) {
      // ---- your current game draw/update ----
      // update_daynight(dt);
      // update agents/mobs, draw_chunks/resources/mobs/player etc

      // quick save hotkey
      if (IsKeyPressed(KEY_F5))
        save_world_to_disk(g_world_name);
      if (IsKeyPressed(KEY_P))
        g_state = STATE_PAUSED;
    } else if (g_state == STATE_TITLE) {

      DrawText("SAMCRAFT", 40, 40, 52, RAYWHITE);
      DrawText("F5 = Save while playing", 44, 100, 18,
               (Color){200, 200, 200, 180});

      Rectangle b1 = (Rectangle){60, 160, 260, 50};
      Rectangle b2 = (Rectangle){60, 220, 260, 50};
      Rectangle b3 = (Rectangle){60, 280, 260, 50};

      if (ui_button(b1, "Play (Load/Select)"))
        g_state = STATE_WORLD_SELECT;
      if (ui_button(b2, "Create World"))
        g_state = STATE_WORLD_CREATE;
      if (ui_button(b3, "Quit"))
        CloseWindow();
    } else if (g_state == STATE_WORLD_CREATE) {

      DrawText("Create World", 60, 50, 34, RAYWHITE);

      DrawText("World Name", 60, 120, 18, RAYWHITE);
      ui_textbox((Rectangle){60, 145, 360, 45}, g_world_name,
                 sizeof(g_world_name), &g_typing_name, 0);

      DrawText("Seed", 60, 205, 18, RAYWHITE);
      ui_textbox((Rectangle){60, 230, 200, 45}, g_seed_text,
                 sizeof(g_seed_text), &g_typing_seed, 1);

      if (ui_button((Rectangle){60, 300, 200, 50}, "Create & Play")) {
        g_world_seed = (uint32_t)strtoul(g_seed_text, NULL, 10);
        world_reset(g_world_seed);
        save_world_to_disk(g_world_name); // create initial save
        g_state = STATE_PLAYING;
      }

      if (ui_button((Rectangle){280, 300, 140, 50}, "Back")) {
        g_state = STATE_TITLE;
      }
    } else if (g_state == STATE_WORLD_SELECT) {
      DrawText("Select World", 60, 50, 34, RAYWHITE);
      DrawText("(This screen next: list saves/ folders)", 60, 95, 18,
               (Color){200, 200, 200, 180});

      // For now: quick load the current name
      if (ui_button((Rectangle){60, 140, 260, 50}, "Load World Name")) {
        if (load_world_from_disk(g_world_name))
          load_models_from_disk(g_world_name);
        g_state = STATE_PLAYING;
      }

      if (ui_button((Rectangle){60, 200, 260, 50}, "Back"))
        g_state = STATE_TITLE;
    }

    draw_chunks();
    draw_resources();
    draw_mobs();
    draw_projectiles();
    draw_pickups();

    // bases
    for (int t = 0; t < TRIBE_COUNT; t++) {
      Vector2 bp = world_to_screen(tribes[t].base.position);
      DrawCircleLinesV(bp, tribes[t].base.radius * WORLD_SCALE,
                       tribes[t].color);
    }

    // agents
    for (int i = 0; i < MAX_AGENTS; i++) {
      if (!agents[i].alive)
        continue;
      Vector2 ap = world_to_screen(agents[i].position);
      Color tc = tribes[agents[i].agent_id / AGENT_PER_TRIBE].color;
      draw_agent(&agents[i], ap, tc);
    }

    // player
    Vector2 pp = world_to_screen(player.position);
    draw_player(pp);
    draw_bow_charge_fx();

    // UI + debug
    draw_ui();
    draw_hover_label();
    draw_minimap();
    draw_daynight_overlay(); // AFTER world draw, before EndDrawing
    draw_hurt_vignette();
    draw_crafting_ui();

    DrawText("MUZE Tribal Simulation", 20, 160, 20, RAYWHITE);
    DrawText(TextFormat("FPS: %d", GetFPS()), 20, 185, 20, RAYWHITE);

    EndDrawing();
  }

  stop_workers();

  CloseWindow();
  return 0;
}
