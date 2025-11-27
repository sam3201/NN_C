#include "mcts.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

/* Node structure */
typedef struct Node {
    float *W;   /* sum of values per action */
    float *Q;   /* mean value per action */
    int   *N;   /* visit counts */
    float *P;   /* prior probs */

    struct Node **children;
    float *latent; /* latent state stored at node */

    int action_count;
    int expanded;
} Node;

/* allocate node */
static Node *node_create(int action_count, int latent_dim) {
    Node *n = (Node*)calloc(1, sizeof(Node));
    if (!n) return NULL;
    n->action_count = action_count;
    n->W = (float*)calloc(action_count, sizeof(float));
    n->Q = (float*)calloc(action_count, sizeof(float));
    n->N = (int*)calloc(action_count, sizeof(int));
    n->P = (float*)calloc(action_count, sizeof(float));
    n->children = (Node**)calloc(action_count, sizeof(Node*));
    n->latent = (float*)calloc(latent_dim, sizeof(float));
    n->expanded = 0;
    return n;
}

/* free node recursively */
static void node_free(Node *n) {
    if (!n) return;
    if (n->W) free(n->W);
    if (n->Q) free(n->Q);
    if (n->N) free(n->N);
    if (n->P) free(n->P);
    if (n->latent) free(n->latent);
    if (n->children) {
        for (int i=0;i<n->action_count;i++) if (n->children[i]) node_free(n->children[i]);
        free(n->children);
    }
    free(n);
}

/* sum visits */
static int node_Nsum(Node *n) {
    int s = 0;
    for (int i=0;i<n->action_count;i++) s += n->N[i];
    return s;
}

/* simple softmax */
static void softmax(const float *logits, int len, float *out) {
    float maxv = -INFINITY;
    for (int i=0;i<len;i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i=0;i<len;i++) {
        float e = expf(logits[i] - maxv);
        out[i] = e; sum += e;
    }
    if (sum <= 0.0f) {
        for (int i=0;i<len;i++) out[i] = 1.0f / (float)len;
    } else {
        for (int i=0;i<len;i++) out[i] /= sum;
    }
}

/* PUCT selection */
static int select_puct(Node *n, float c_puct) {
    int best = 0;
    float best_score = -FLT_MAX;
    int Nsum = node_Nsum(n);
    float sqrt_N = sqrtf((float)(Nsum + 1));
    for (int a=0;a<n->action_count;a++) {
        float Q = n->Q[a];
        float P = n->P[a];
        float U = c_puct * P * (sqrt_N / (1.0f + (float)n->N[a]));
        float score = Q + U;
        if (score > best_score) { best_score = score; best = a; }
    }
    return best;
}

/* Add Dirichlet noise to root priors */
static void add_dirichlet_noise(Node *root, float alpha, float eps) {
    if (!root || alpha <= 0.0f || eps <= 0.0f) return;
    int A = root->action_count;
    /* sample gamma via -ln(U) trick (exponential) then normalize -> Dirichlet */
    float *g = (float*)malloc(sizeof(float)*A);
    float sum = 0.0f;
    for (int i=0;i<A;i++) {
        float u = (rand()+1.0f) / (RAND_MAX + 1.0f);
        g[i] = -logf(u);
        sum += g[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (int i=0;i<A;i++) {
        float d = g[i] / sum; /* Dirichlet sample */
        root->P[i] = (1.0f - eps) * root->P[i] + eps * d;
    }
    free(g);
}

/* Expand node: use model.predict on node->latent to fill P and return predicted value (caller can use) */
static float expand_node(Node *node, MuModel *model) {
    int A = node->action_count;
    float *logits = (float*)malloc(sizeof(float)*A);
    float value = 0.0f;
    mu_model_predict(model, node->latent, logits, &value);
    /* fill P via softmax */
    float *pri = (float*)malloc(sizeof(float)*A);
    softmax(logits, A, pri);
    for (int a=0;a<A;a++) node->P[a] = pri[a];
    node->expanded = 1;
    free(pri);
    free(logits);
    return value;
}

/* Backup with discounted sum:
   Given actions[0..depth-1], rewards[0..depth-1] (each reward r_t is reward received when taking actions[t])
   and leaf_value v (prediction at leaf), compute total_return = sum_{i=0..depth-1} gamma^i * r_i + gamma^depth * v
   Then add total_return to W of the root action and ancestors appropriately.
   We implement: for each ancestor along path, add total_return to W[action_at_that_level], increment N, recompute Q.
*/
static void backup_with_discount(Node *root, int *actions, float *rewards, int depth, float leaf_value, float gamma) {
    /* compute discounted return for entire path from root */
    float total = 0.0f;
    float gpow = 1.0f;
    for (int i=0;i<depth;i++) {
        total += gpow * rewards[i];
        gpow *= gamma;
    }
    total += gpow * leaf_value; /* gamma^depth * v */

    /* now walk root and update each action's stats along the path */
    Node *n = root;
    for (int i=0;i<depth;i++) {
        int a = actions[i];
        n->W[a] += total;
        n->N[a] += 1;
        n->Q[a] = n->W[a] / (float)n->N[a];
        if (!n->children[a]) return;
        n = n->children[a];
    }
}

/* convert visit counts to policy pi with temperature */
static void visits_to_pi(Node *root, float temperature, float *pi_out) {
    int A = root->action_count;
    if (temperature <= 0.0f) temperature = 1e-6f;
    double sum = 0.0;
    for (int a=0;a<A;a++) {
        double cnt = (double)root->N[a];
        double val = pow(cnt, 1.0 / (double)temperature);
        if (isnan(val) || isinf(val)) val = 0.0;
        pi_out[a] = (float)val;
        sum += val;
    }
    if (sum <= 0.0) {
        for (int a=0;a<A;a++) pi_out[a] = 1.0f / (float)A;
    } else {
        for (int a=0;a<A;a++) pi_out[a] /= (float)sum;
    }
}

/* Main MCTS run */
MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params) {
    MCTSResult res = {0, NULL, 0, 0.0f};
    if (!model || !obs || !params) return res;

    int A = model->cfg.action_count;
    int L = model->cfg.latent_dim;

    Node *root = node_create(A, L);
    if (!root) return res;

    /* root latent */
    mu_model_repr(model, obs, root->latent);
    /* expand root and get root value (unused directly) */
    float root_value = expand_node(root, model);

    /* optional dirichlet noise on root priors */
    if (params->dirichlet_alpha > 0.0f && params->dirichlet_eps > 0.0f) {
        add_dirichlet_noise(root, params->dirichlet_alpha, params->dirichlet_eps);
    }

    /* arrays for traversal */
    int max_depth = params->max_depth > 0 ? params->max_depth : 64;
    int *actions = (int*)malloc(sizeof(int) * max_depth);
    float *rewards = (float*)malloc(sizeof(float) * max_depth);

    for (int sim=0; sim<params->num_simulations; ++sim) {
        Node *node = root;
        int depth = 0;
        float *h_cur = (float*)malloc(sizeof(float) * L);
        memcpy(h_cur, root->latent, sizeof(float) * L);

        while (node->expanded && depth < max_depth) {
            int a = select_puct(node, params->c_puct);
            actions[depth] = a;
            /* compute dynamics -> child latent and reward */
            float *h_next = NULL;
            if (!node->children[a]) {
                /* create child, fill its latent by dynamics */
                node->children[a] = node_create(A, L);
                h_next = node->children[a]->latent;
                float r = 0.0f;
                mu_model_dynamics(model, h_cur, a, h_next, &r);
                rewards[depth] = r;
                /* expand child and get its predicted value */
                float leaf_value = expand_node(node->children[a], model);
                /* backup using rewards along path + leaf_value */
                backup_with_discount(root, actions, rewards, depth+1, leaf_value, params->discount);
                /* finished this simulation */
                free(h_cur);
                break;
            } else {
                /* child exists: advance via dynamics but also update child's latent */
                h_next = node->children[a]->latent;
                float r = 0.0f;
                mu_model_dynamics(model, h_cur, a, h_next, &r);
                rewards[depth] = r;
                /* step into child */
                node = node->children[a];
                /* copy h_next into h_cur for further steps */
                memcpy(h_cur, h_next, sizeof(float) * L);
                depth++;
                continue;
            }
        } /* end traverse */

        free(h_cur);
    } /* end sims */

    /* compute pi */
    float *pi = (float*)malloc(sizeof(float) * A);
    visits_to_pi(root, params->temperature, pi);

    /* choose best action by highest Q (if N>0), else highest prior */
    int best_a = 0;
    float best_q = -INFINITY;
    for (int a=0;a<A;a++) {
        float q = (root->N[a] > 0) ? root->Q[a] : -INFINITY;
        if (q > best_q) { best_q = q; best_a = a; }
    }
    /* if all N==0, choose max prior */
    if (best_q == -INFINITY) {
        float best_p = -INFINITY;
        for (int a=0;a<A;a++) {
            if (root->P[a] > best_p) { best_p = root->P[a]; best_a = a; }
        }
    }

    res.action_count = A;
    res.pi = pi;
    res.chosen_action = best_a;
    res.root_value = root_value;

    /* cleanup */
    free(actions);
    free(rewards);
    node_free(root);
    return res;
}

void mcts_result_free(MCTSResult *res) {
    if (!res) return;
    if (res->pi) free(res->pi);
    res->pi = NULL;
    res->action_count = 0;
    res->chosen_action = 0;
    res->root_value = 0.0f;
}

