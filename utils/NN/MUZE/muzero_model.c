#include "muzero_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*
  Simple MuZero model wrapper.
  - This is a skeleton: plug your NN_C forward/backward/update here.
  - For now it uses tiny random-initialized linear layers inside the struct as placeholders.
*/

/* helpers */
static float randf() { return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f; }

MuModel *mu_model_create(const MuConfig *cfg) {
    MuModel *m = (MuModel*)calloc(1, sizeof(MuModel));
    if (!m) return NULL;
    m->cfg = *cfg;
    size_t obs = (size_t)cfg->obs_dim;
    size_t lat = (size_t)cfg->latent_dim;
    size_t act = (size_t)cfg->action_count;

    /* allocate placeholder params */
    m->repr_W = (float*)malloc(obs * lat * sizeof(float));
    m->dynamics_W = (float*)malloc((lat + act) * lat * sizeof(float));
    m->reward_W = (float*)malloc((lat + act) * 1 * sizeof(float));
    m->policy_W = (float*)malloc(lat * act * sizeof(float));
    m->value_W = (float*)malloc(lat * 1 * sizeof(float));

    if (!m->repr_W || !m->dynamics_W || !m->reward_W || !m->policy_W || !m->value_W) {
        mu_model_free(m);
        return NULL;
    }

    /* random init small */
    for (size_t i=0;i<obs*lat;i++) m->repr_W[i] = randf()*0.01f;
    for (size_t i=0;i<(lat+act)*lat;i++) m->dynamics_W[i] = randf()*0.01f;
    for (size_t i=0;i<(lat+act);i++) m->reward_W[i] = 0.0f;
    for (size_t i=0;i<lat*act;i++) m->policy_W[i] = randf()*0.01f;
    for (size_t i=0;i<lat;i++) m->value_W[i] = randf()*0.01f;

    return m;
}

void mu_model_free(MuModel *m) {
    if (!m) return;
    free(m->repr_W);
    free(m->dynamics_W);
    free(m->reward_W);
    free(m->policy_W);
    free(m->value_W);
    free(m);
}

/* Tiny linear helpers (placeholder). Replace with real NN forward calls. */
static void linear_mul(const float *W, int in_dim, int out_dim, const float *x, float *y) {
    /* y = W^T x  with W stored row-major as [in_dim x out_dim] */
    for (int j=0;j<out_dim;j++) {
        float sum = 0.0f;
        for (int i=0;i<in_dim;i++) {
            sum += W[i*out_dim + j] * x[i];
        }
        y[j] = sum;
    }
}
static void relu_inplace(float *v, int n) {
    for (int i=0;i<n;i++) if (v[i] < 0.0f) v[i] = 0.0f;
}

/* Representation: obs -> h */
void mu_model_repr(MuModel *m, const float *obs, float *h_out) {
    int od = m->cfg.obs_dim;
    int hd = m->cfg.latent_dim;
    /* placeholder linear */
    linear_mul(m->repr_W, od, hd, obs, h_out);
    /* simple nonlinearity */
    relu_inplace(h_out, hd);

    /* TODO: Replace with your NN_C representation forward: f_theta(obs) -> h_out */
}

/* Dynamics: (h_in, action) -> h_out, reward_out */
void mu_model_dynamics(MuModel *m, const float *h_in, int action, float *h_out, float *reward_out) {
    int hd = m->cfg.latent_dim;
    int act = m->cfg.action_count;
    /* build input vector [h_in ; onehot(action)] */
    float *buf = (float*)malloc((hd + act) * sizeof(float));
    if (!buf) return;
    for (int i=0;i<hd;i++) buf[i] = h_in[i];
    for (int a=0;a<act;a++) buf[hd + a] = (a==action) ? 1.0f : 0.0f;

    linear_mul(m->dynamics_W, hd + act, hd, buf, h_out);
    relu_inplace(h_out, hd);

    /* reward */
    float r = 0.0f;
    linear_mul(m->reward_W, hd + act, 1, buf, &r);
    *reward_out = r;

    free(buf);

    /* TODO: Replace with your NN_C dynamics forward: g_theta(h, a) -> h_next ; r_theta(h,a) -> reward */
}

/* Prediction: h -> policy_logits (len action_count), value (scalar out) */
void mu_model_predict(MuModel *m, const float *h, float *policy_logits_out, float *value_out) {
    int hd = m->cfg.latent_dim;
    int act = m->cfg.action_count;

    linear_mul(m->policy_W, hd, act, h, policy_logits_out);
    /* optionally scale logits */
    linear_mul(m->value_W, hd, 1, h, value_out);

    /* TODO: Replace with your NN_C prediction forward: o_theta(h) -> policy_logits, value */
}

/* Save / load stubs: integrate with your project's serialization */
int mu_model_save(MuModel *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&m->cfg, sizeof(MuConfig), 1, f);
    size_t obs = m->cfg.obs_dim, lat = m->cfg.latent_dim, act = m->cfg.action_count;
    fwrite(m->repr_W, sizeof(float), obs*lat, f);
    fwrite(m->dynamics_W, sizeof(float), (lat+act)*lat, f);
    fwrite(m->reward_W, sizeof(float), (lat+act), f);
    fwrite(m->policy_W, sizeof(float), lat*act, f);
    fwrite(m->value_W, sizeof(float), lat, f);
    fclose(f);
    return 0;
}
int mu_model_load(MuModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    MuConfig cfg_disk;
    fread(&cfg_disk, sizeof(MuConfig), 1, f);
    if (cfg_disk.obs_dim != m->cfg.obs_dim || cfg_disk.latent_dim != m->cfg.latent_dim || cfg_disk.action_count != m->cfg.action_count) {
        fclose(f);
        return -2;
    }
    size_t obs = m->cfg.obs_dim, lat = m->cfg.latent_dim, act = m->cfg.action_count;
    fread(m->repr_W, sizeof(float), obs*lat, f);
    fread(m->dynamics_W, sizeof(float), (lat+act)*lat, f);
    fread(m->reward_W, sizeof(float), (lat+act), f);
    fread(m->policy_W, sizeof(float), lat*act, f);
    fread(m->value_W, sizeof(float), lat, f);
    fclose(f);
    return 0;
}

