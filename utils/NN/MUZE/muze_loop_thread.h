#ifndef MUZE_LOOP_THREAD_H
#define MUZE_LOOP_THREAD_H

#include "muze_env.h"
#include "muze_loop.h"
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  pthread_t thread;
  int running;
  int stop;
  int use_multi;

  MuModel *model;
  MuzeEnv env;
  ReplayBuffer *rb;
  GameReplay *gr;
  MCTSParams mcts;
  SelfPlayParams selfplay;
  MuLoopConfig loop;
  MCTSRng rng;

  pthread_mutex_t *model_mutex;
  pthread_mutex_t *rb_mutex;
  pthread_mutex_t *gr_mutex;
} MuzeLoopThread;

int muze_loop_thread_start(MuzeLoopThread *loop);
void muze_loop_thread_stop(MuzeLoopThread *loop);

#ifdef __cplusplus
}
#endif

#endif
