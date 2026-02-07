#include "muze_config_util.h"
#include "../NN/NN.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void muze_config_init_defaults(MuzeConfig *cfg, int obs_dim, int action_count) {
  if (!cfg)
    return;
  memset(cfg, 0, sizeof(*cfg));

  cfg->model.obs_dim = obs_dim;
  cfg->model.latent_dim = 64;
  cfg->model.action_count = action_count;

  cfg->nn.opt_repr = ADAM;
  cfg->nn.opt_dyn = ADAM;
  cfg->nn.opt_pred = ADAM;
  cfg->nn.opt_vprefix = ADAM;
  cfg->nn.opt_reward = ADAM;

  cfg->nn.loss_repr = MSE;
  cfg->nn.loss_dyn = MSE;
  cfg->nn.loss_pred = MSE;
  cfg->nn.loss_vprefix = MSE;
  cfg->nn.loss_reward = MSE;
  cfg->nn.lossd_repr = MSE_DERIVATIVE;
  cfg->nn.lossd_dyn = MSE_DERIVATIVE;
  cfg->nn.lossd_pred = MSE_DERIVATIVE;
  cfg->nn.lossd_vprefix = MSE_DERIVATIVE;
  cfg->nn.lossd_reward = MSE_DERIVATIVE;

  cfg->nn.lr_repr = 0.001L;
  cfg->nn.lr_dyn = 0.001L;
  cfg->nn.lr_pred = 0.001L;
  cfg->nn.lr_vprefix = 0.001L;
  cfg->nn.lr_reward = 0.001L;

  cfg->nn.hidden_repr = 128;
  cfg->nn.hidden_dyn = 128;
  cfg->nn.hidden_pred = 128;
  cfg->nn.hidden_vprefix = 128;
  cfg->nn.hidden_reward = 128;

  cfg->nn.use_value_support = 1;
  cfg->nn.use_reward_support = 1;
  cfg->nn.support_size = 21;
  cfg->nn.support_min = -2.0f;
  cfg->nn.support_max = 2.0f;
  cfg->nn.action_embed_dim = 64;
  cfg->nn.grad_clip = 5.0f;
  cfg->nn.global_grad_clip = 1.0f;

  cfg->mcts.num_simulations = 80;
  cfg->mcts.batch_simulations = 8;
  cfg->mcts.c_puct = 1.25f;
  cfg->mcts.max_depth = 16;
  cfg->mcts.dirichlet_alpha = 0.3f;
  cfg->mcts.dirichlet_eps = 0.25f;
  cfg->mcts.temperature = 1.0f;
  cfg->mcts.discount = 0.997f;

  cfg->selfplay.max_steps = 200;
  cfg->selfplay.gamma = 0.997f;
  cfg->selfplay.temp_start = 1.0f;
  cfg->selfplay.temp_end = 0.25f;
  cfg->selfplay.temp_decay_episodes = 200;
  cfg->selfplay.dirichlet_alpha = 0.3f;
  cfg->selfplay.dirichlet_eps = 0.25f;
  cfg->selfplay.total_episodes = 0;
  cfg->selfplay.log_every = 10;

  cfg->trainer.batch_size = 64;
  cfg->trainer.train_steps = 50;
  cfg->trainer.min_replay_size = 1024;
  cfg->trainer.unroll_steps = 5;
  cfg->trainer.bootstrap_steps = 5;
  cfg->trainer.discount = 0.997f;
  cfg->trainer.use_per = 1;
  cfg->trainer.per_alpha = 0.6f;
  cfg->trainer.per_beta = 0.4f;
  cfg->trainer.per_beta_start = 0.4f;
  cfg->trainer.per_beta_end = 1.0f;
  cfg->trainer.per_beta_anneal_steps = 2000;
  cfg->trainer.per_eps = 1e-3f;
  cfg->trainer.train_reward_head = 0;
  cfg->trainer.reward_target_is_vprefix = 1;
  cfg->trainer.lr = 0.05f;

  cfg->loop.iterations = 1;
  cfg->loop.selfplay_episodes_per_iter = 10;
  cfg->loop.train_calls_per_iter = 1;
  cfg->loop.use_reanalyze = 0;
  cfg->loop.reanalyze_samples_per_iter = 256;
  cfg->loop.reanalyze_gamma = cfg->mcts.discount;
  cfg->loop.reanalyze_full_games = 0;
  cfg->loop.eval_interval = 0;
  cfg->loop.eval_episodes = 0;
  cfg->loop.eval_max_steps = 0;
  cfg->loop.checkpoint_interval = 0;
  cfg->loop.checkpoint_prefix = NULL;
  cfg->loop.checkpoint_save_replay = 0;
  cfg->loop.checkpoint_save_games = 0;
  cfg->loop.checkpoint_keep_last = 0;
  cfg->loop.selfplay_actor_count = 1;
  cfg->loop.selfplay_use_threads = 1;
  cfg->loop.reanalyze_interval = 1;
  cfg->loop.reanalyze_fraction = 0.0f;
  cfg->loop.reanalyze_min_replay = 0;
  cfg->loop.replay_shard_interval = 0;
  cfg->loop.replay_shard_keep_last = 0;
  cfg->loop.replay_shard_max_entries = 0;
  cfg->loop.replay_shard_prefix = NULL;
  cfg->loop.replay_shard_save_games = 0;
  cfg->loop.eval_best_model = 0;
  cfg->loop.best_checkpoint_prefix = NULL;
  cfg->loop.best_save_replay = 0;
  cfg->loop.best_save_games = 0;
  cfg->loop.selfplay_disable = 0;

  cfg->actors.actor_count = 1;
  cfg->actors.use_threads = 1;

  cfg->replay_shards.shard_interval = 0;
  cfg->replay_shards.shard_keep_last = 0;
  cfg->replay_shards.shard_max_entries = 0;
  cfg->replay_shards.shard_prefix = NULL;
  cfg->replay_shards.shard_save_games = 0;

  cfg->reanalyze.interval = 0;
  cfg->reanalyze.fraction = 0.0f;
  cfg->reanalyze.min_replay_size = 0;

  cfg->best.eval_best_model = 0;
  cfg->best.best_checkpoint_prefix = NULL;
  cfg->best.best_save_replay = 0;
  cfg->best.best_save_games = 0;

  cfg->seed.seed = 0;
}
