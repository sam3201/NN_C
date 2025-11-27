#include "../NN/MUZE/muzero_model.h"
#include "../NN/MUZE/mcts.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    MuConfig cfg = { .obs_dim = 8, .latent_dim = 16, .action_count = 5 };
    MuModel *m = mu_model_create(&cfg);
    float obs[8]; for (int i=0;i<8;i++) obs[i] = (float)rand()/RAND_MAX;
    MCTSParams p = { .num_simulations = 50, .c_puct = 1.0f };
    MCTSResult r = mcts_run(m, obs, &p);
    printf("chosen action: %d\n", r.chosen_action);
    printf("pi: "); for (int i=0;i<r.action_count;i++) printf("%f ", r.pi[i]); printf("\n");
    mcts_result_free(&r);
    mu_model_free(m);
    return 0;
}

