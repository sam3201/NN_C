#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../NN/MUZE/muzero_model.h"
#include "../NN/MUZE/mcts.h"

/*
    This test verifies ALL functionality:
    - MuModel creation
    - Representation f(o) -> h
    - Dynamics g(h,a) -> h', r
    - Prediction h -> (policy logits, value)
    - MCTS rollout
*/

/* Helper: print vector */
static void print_vec(const char *name, const float *v, int n)
{
    printf("%s: [", name);
    for (int i = 0; i < n; i++)
    {
        printf("%g", v[i]);
        if (i + 1 < n) printf(", ");
    }
    printf("]\n");
}

int main(void)
{
    printf("==== MuZero TEST START ====\n");

    /* 1) CONFIGURE THE MUZERO MODEL */
    MuConfig cfg = {
        .obs_dim = 4,
        .latent_dim = 8,
        .action_count = 3
    };

    MuModel *model = mu_model_create(&cfg);
    if (!model) {
        printf("FATAL: mu_model_create failed.\n");
        return 1;
    }

    printf("Model created. obs_dim=%d latent_dim=%d action_count=%d\n",
        cfg.obs_dim, cfg.latent_dim, cfg.action_count);


    /* 2) TEST REPRESENTATION */
    float obs[4] = {1.0f, 0.5f, -0.25f, 0.1f};
    float latent[8];

    mu_model_repr(model, obs, latent);
    print_vec("Representation latent", latent, cfg.latent_dim);


    /* 3) TEST PREDICTION */
    float policy_logits[3];
    float value;

    mu_model_predict(model, latent, policy_logits, &value);
    print_vec("Policy logits", policy_logits, cfg.action_count);
    printf("Value: %f\n", value);


    /* 4) TEST DYNAMICS */
    float latent2[8];
    float reward;

    mu_model_dynamics(model, latent, 1, latent2, &reward);
    print_vec("Dynamics latent2", latent2, cfg.latent_dim);
    printf("Reward: %f\n", reward);


    /* 5) TEST MCTS */
    MCTSParams mp = {
        .num_simulations = 25,
        .c_puct = 1.25f,
        .max_depth = 10,

        .dirichlet_alpha = 0.3f,
        .dirichlet_eps   = 0.25f,

        .temperature = 1.0f,
        .discount = 0.997f
    };

    printf("\n=== Running MCTS ===\n");
    MCTSResult res = mcts_run(model, obs, &mp);

    printf("MCTS chosen_action = %d\n", res.chosen_action);
    print_vec("MCTS pi", res.pi, cfg.action_count);

    mcts_result_free(&res);


    /* 6) CLEANUP */
    mu_model_free(model);

    printf("==== MuZero TEST END ====\n");
    return 0;
}

