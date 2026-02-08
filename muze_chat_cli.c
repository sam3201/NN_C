#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Remove non-existent includes - using built-in functionality only
#include <string.h>
#include <stdint.h>

static uint32_t djb2_hash(const char *s) {
    uint32_t h = 5381u;
    int c;
    while ((c = *s++)) {
        h = ((h << 5) + h) + (uint32_t)c;
    }
    return h;
}

static void text_to_raw_obs4(const char *text, float raw_obs[4]) {
    uint32_t h = djb2_hash(text ? text : "");
    raw_obs[0] = 1.0f;
    raw_obs[1] = (float)((h % 1000u) / 1000.0);
    raw_obs[2] = (float)((h >> 10) & 0xFFu);
    raw_obs[3] = (float)((h >> 18) & 0x3FFFu);
}

static int json_get_string(const char *json, const char *key, char *out, size_t out_sz) {
    if (!json || !key || !out || out_sz == 0) return 0;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < out_sz) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return i > 0;
}

static int json_get_float(const char *json, const char *key, float *out) {
    if (!json || !key || !out) return 0;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    *out = (float)strtod(p, NULL);
    return 1;
}

static int json_get_int(const char *json, const char *key, int *out) {
    if (!json || !key || !out) return 0;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    *out = (int)strtol(p, NULL, 10);
    return 1;
}

static char *read_all_stdin(void) {
    size_t cap = 8192;
    size_t n = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        if (n + 4096 + 1 > cap) {
            cap *= 2;
            char *nb = (char *)realloc(buf, cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        size_t got = fread(buf + n, 1, 4096, stdin);
        n += got;
        if (got == 0) break;
    }
    buf[n] = '\0';
    return buf;
}

int main(void) {
    char *json = read_all_stdin();
    if (!json) {
        printf("{\"ok\":false,\"error\":\"read_stdin_failed\"}\n");
        return 1;
    }

    char model_path[512];
    model_path[0] = '\0';
    if (!json_get_string(json, "model_path", model_path, sizeof(model_path))) {
        snprintf(model_path, sizeof(model_path), "MUZE_STATE/muze_enhanced.bin");
    }

    char text[2048];
    text[0] = '\0';
    if (!json_get_string(json, "text", text, sizeof(text))) {
        snprintf(text, sizeof(text), "");
    }

    float reward = 0.0f;
    int done = 0;
    int do_train = 0;
    (void)json_get_float(json, "reward", &reward);
    (void)json_get_int(json, "done", &done);
    (void)json_get_int(json, "train", &do_train);

    MuzeEnhancedConfig cfg;
    init_enhanced_config(&cfg);
    cfg.use_discrete_actions = 0;
    cfg.use_continuous_actor = 1;
    cfg.pack_actions = 0;

    MuzeEnhancedModel *model = NULL;
    FILE *fp = fopen(model_path, "rb");
    if (fp) {
        fclose(fp);
        model = muze_enhanced_model_load(model_path);
    }
    if (!model) {
        model = muze_enhanced_model_create(&cfg);
    }

    if (!model) {
        free(json);
        printf("{\"ok\":false,\"error\":\"model_create_failed\"}\n");
        return 2;
    }

    float raw_obs[4];
    text_to_raw_obs4(text, raw_obs);
    CompressedObs obs;
    init_compressed_obs(&obs);
    compress_observation(raw_obs, 4, &obs);

    int action = 0;
    float value = 0.0f;
    float policy[4] = {0};
    muze_enhanced_model_forward(model, &obs, &action, &value, policy);

    if (do_train) {
        muze_enhanced_model_train_step(model, &obs, action, reward, done);
        muze_enhanced_model_update_weights(model);
        (void)muze_enhanced_model_save(model, model_path);
    }

    printf("{\"ok\":true,\"action\":%d,\"value\":%.6f,\"loss\":%.6f,\"training_step\":%d,\"model_path\":\"%s\"}\n",
           action, value, model->loss_value, model->training_step, model_path);

    muze_enhanced_model_destroy(model);
    free(json);
    return 0;
}
