// runtime.h
runtime.h : runtime.h : #ifndef MUZE_RUNTIME_H runtime
                            .h : #define MUZE_RUNTIME_H runtime.h
    : runtime.h : #include
                  "muze_cortex.h" runtime.h
    : #include "muzero_model.h" runtime.h
    : #include "replay_buffer.h" runtime.h
    : #include "trainer.h" runtime.h : #include<stdint.h>
                                           runtime.h
    : runtime.h
    : #define TRAIN_WINDOW 1024 // training cache size, NOT memory size
      runtime.h : #define TRAIN_WARMUP TRAIN_WINDOW // warmup cache size
                      runtime.h : typedef struct {
  runtime.h : ReplayBuffer *rb;
  runtime.h : runtime.h : float *last_obs;
  runtime.h : float *last_pi;
  runtime.h : int last_action;
  runtime.h : int has_last;
  runtime.h : runtime.h : float gamma;
  runtime.h : runtime
                  .h : /* infinite logical memory */
                       runtime.h : size_t total_steps;
  runtime.h : runtime.h : TrainerConfig cfg;
  runtime.h : bool has_cfg;
  runtime.h:
} MuRuntime;
runtime.h : runtime
                .h : /* Runtime lifecycle */
                     runtime.h : MuRuntime *
                                 mu_runtime_create(MuModel *model, float gamma);
runtime.h : void mu_runtime_free(MuRuntime *rt);
runtime.h : runtime
                .h : /* : set/get trainer config */
                     runtime.h
    : void
      mu_runtime_set_trainer_config(MuRuntime *rt, const TrainerConfig *cfg);
runtime.h : TrainerConfig mu_runtime_get_trainer_config(const MuRuntime *rt);
runtime.h : runtime
                .h : /* Runtime operations (internal) */
                     runtime.h
    : void
      mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                      runtime.h : int action, float reward);
runtime.h : runtime.h
    : void
      mu_runtime_step_with_pi(MuRuntime *rt, MuModel *model, const float *obs,
                              runtime.h : const float *pi, int action,
                              float reward);
runtime.h : runtime.h
    : void
      mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                             runtime.h : float terminal_reward);
runtime.h : runtime.h : void mu_runtime_reset_episode(MuRuntime *rt);
runtime.h : void mu_runtime_train(MuRuntime *rt, MuModel *model,
                                  const TrainerConfig *cfg);
runtime.h : runtime.h : int muze_select_action(MuCortex *cortex,
                                               const float *obs, size_t obs_dim,
                                               runtime.h : float *out_pi,
                                               size_t action_count,
                                               MCTSRng *rng);
runtime.h : runtime.h : #endif
