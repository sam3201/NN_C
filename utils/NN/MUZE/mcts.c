#include "mcts.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <time.h>

/* Simple MCTS node */
typedef struct Node {
    /* statistics */
    float *W; /* sum of values per action */
    float *Q; /* mean value per action */
    int *N;   /* visit counts per action */
    float *P; /* prior probabilities per action */

    /* children pointers */
    struct Node **children; /* array of pointers to child nodes (len action_count) */

    int action_count;
    int expanded; /* bool */
    /* store latent state optionally for MuZero (not stored in this simple implementation) */
} Node;

static Node *node_create(int action_count) {
    Node *n = (Node*)calloc(1, sizeof(Node));
    n->action_count = action_count;
    n->W = (float*)calloc(action_count, sizeof(float));
    n->Q = (float*)calloc(action_count, sizeof(float));
    n->N = (int*)calloc(action_count, sizeof(int));
    n->P = (float*)calloc(action_count, sizeof(float));
    n->children = (Node**)calloc(action_count, sizeof(Node*));
    n->expanded = 0;
    return n;
}
static void node_free(Node *n) {
    if (!n) return;
    free(n->W); free(n->Q); free(n->N); free(n->P);
    if (n->children) {
        for (int i=0;i<n->action_count;i++) node_free(n->children[i]);
        free(n->children);
    }
    free(n);
}

/* PUCT selection */
static int select_puct(Node *n, float c_puct) {
    int best_a = 0;
    float best_score = -FLT_MAX;
    int Nsum = 0;
    for (int i=0;i<n->action_count;i++) Nsum += n->N[i];
    for (int a=0;a<n->action_count;a++) {
        float Q = n->Q[a];
        float P = n->P[a];
        float U = c_puct * P * sqrtf((float)(Nsum + 1)) / (1.0f + (float)n->N[a]);
        float score = Q + U;
        if (score > best_score) { best_score = score; best_a = a; }
    }
    return best_a;
}

/* backup value up a path (simple: add v to W, increment N, recompute Q) */
static void backup(Node *node, int *actions, int depth, float value) {
    /* actions is path of length depth (actions[0] is first action from root) */
    Node *n = node;
    for (int i=0;i<depth;i++) {
        int a = actions[i];
        n->W[a] += value;
        n->N[a] += 1;
        n->Q[a] = n->W[a] / (float)n->N[a];
        if (!n->children[a]) return; /* ended early */
        n = n->children[a];
    }
}

/* Expand a leaf node using model: fill P (softmax of logits) and mark expanded */
static void expand_node_with_model(Node *node, MuModel *model, const float *latent_h) {
    /* This implementation expects model->predict to produce logits and value, we only use logits -> P here */
    int A = node->action_count;
    float *logits = (float*)malloc(sizeof(float) * A);
    float v;
    /* NOTE: mu_model_predict expects latent h. In this simplified MCTS we do not store latent per node.
       In a full MuZero implementation you must call mu_model_dynamics during traversal and keep latent states. */
    mu_model_predict(model, latent_h, logits, &v);

    /* softmax */
    float maxl = -INFINITY;
    for (int i=0;i<A;i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0.0f;
    for (int i=0;i<A;i++) {
        logits[i] = expf(logits[i] - maxl);
        sum += logits[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (int i=0;i<A;i++) node->P[i] = logits[i] / sum;

    node->expanded = 1;
    free(logits);
}

/* Compute pi from root visit counts and choose action (argmax) */
static void compute_root_policy(Node *root, float *pi_out) {
    int A = root->action_count;
    int total = 0;
    for (int a=0;a<A;a++) total += root->N[a];
    if (total == 0) total = 1;
    for (int a=0;a<A;a++) {
        pi_out[a] = (float)root->N[a] / (float)total;
    }
}

/* Simple interface function: runs MCTS using the mu_model
   NOTE: This is a simplified version: it uses the representation only at root and calls dynamics
   on-the-fly but does not store latent states per node. For correct MuZero you must store latent per node.
*/
MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params) {
    int A = model->cfg.action_count;
    Node *root = node_create(A);

    /* compute root latent h0 */
    float *h0 = (float*)malloc(sizeof(float) * model->cfg.latent_dim);
    mu_model_repr(model, obs, h0);

    /* expand root once */
    expand_node_with_model(root, model, h0);

    /* arrays for traversal */
    int max_depth = 64;
    int *actions = (int*)malloc(sizeof(int) * max_depth);

    /* Monte Carlo simulations */
    for (int sim=0; sim < params->num_simulations; ++sim) {
        /* traverse */
        Node *node = root;
        float *h_cur = (float*)malloc(sizeof(float) * model->cfg.latent_dim);
        memcpy(h_cur, h0, sizeof(float) * model->cfg.latent_dim);
        int depth = 0;
        while (node->expanded && depth < max_depth) {
            int a = select_puct(node, params->c_puct);
            actions[depth++] = a;

            /* get next latent via dynamics */
            float *h_next = (float*)malloc(sizeof(float) * model->cfg.latent_dim);
            float r = 0.0f;
            mu_model_dynamics(model, h_cur, a, h_next, &r);

            /* ensure child exists */
            if (!node->children[a]) node->children[a] = node_create(A);
            node = node->children[a];

            free(h_cur);
            h_cur = h_next;

            if (!node->expanded) {
                /* expand leaf with model.predict using h_cur */
                expand_node_with_model(node, model, h_cur);
                /* get value from predict for backup
                   mu_model_predict returns value; but expand_node_with_model didn't capture it.
                   Call predict again to get value (inefficient but simple). */
                float *dummy_logits = (float*)malloc(sizeof(float)*A);
                float value = 0.0f;
                mu_model_predict(model, h_cur, dummy_logits, &value);
                free(dummy_logits);

                /* backup value up the path */
                backup(root, actions, depth, value);
                free(h_cur);
                break;
            }
            /* else continue down */
        }
        /* end sim */
    }

    /* create output pi and choose best action */
    float *pi = (float*)malloc(sizeof(float) * A);
    compute_root_policy(root, pi);
    int best_action = 0;
    float best_val = -INFINITY;
    for (int a=0;a<A;a++) {
        if (root->N[a] > 0 && root->Q[a] > best_val) {
            best_val = root->Q[a];
            best_action = a;
        }
    }

    /* cleanup */
    node_free(root);
    free(h0);
    free(actions);

    MCTSResult res;
    res.action_count = A;
    res.pi = pi;
    res.chosen_action = best_action;
    return res;
}

void mcts_result_free(MCTSResult *res) {
    if (!res) return;
    if (res->pi) free(res->pi);
    res->pi = NULL;
}

