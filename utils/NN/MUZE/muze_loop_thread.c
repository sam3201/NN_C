#include "muze_loop_thread.h"
#include <string.h>

static void *muze_loop_thread_main(void *arg) {
  MuzeLoopThread *loop = (MuzeLoopThread *)arg;
  if (!loop)
    return NULL;

  loop->running = 1;

  int remaining = loop->loop.iterations;
  int infinite = (remaining <= 0);

  while (!loop->stop) {
    MuLoopConfig cfg = loop->loop;
    cfg.iterations = 1;

    if (loop->use_multi) {
      muze_run_loop_multi_locked(loop->model, loop->env.state, loop->env.reset,
                                 loop->env.step, loop->env.clone,
                                 loop->env.destroy, loop->rb, loop->gr,
                                 &loop->mcts, &loop->selfplay, &cfg,
                                 &loop->rng, loop->model_mutex, loop->rb_mutex,
                                 loop->gr_mutex);
    } else {
      muze_run_loop_locked(loop->model, loop->env.state, loop->env.reset,
                           loop->env.step, loop->rb, loop->gr, &loop->mcts,
                           &loop->selfplay, &cfg, &loop->rng,
                           loop->model_mutex, loop->rb_mutex, loop->gr_mutex);
    }

    if (!infinite) {
      remaining--;
      if (remaining <= 0)
        break;
    }
  }

  loop->running = 0;
  return NULL;
}

int muze_loop_thread_start(MuzeLoopThread *loop) {
  if (!loop || loop->running)
    return -1;
  loop->stop = 0;
  if (pthread_create(&loop->thread, NULL, muze_loop_thread_main, loop) != 0)
    return -1;
  return 0;
}

void muze_loop_thread_stop(MuzeLoopThread *loop) {
  if (!loop || !loop->running)
    return;
  loop->stop = 1;
  pthread_join(loop->thread, NULL);
  loop->running = 0;
}
