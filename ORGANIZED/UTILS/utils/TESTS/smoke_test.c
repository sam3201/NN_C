#include "../NN/NEAT.h"
#include "../NN/NN.h"
#include "../NN/TRANSFORMER.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void smoke_test_nn(void) {
  printf("== NN smoke test ==\n");

  size_t layers[] = {2, 3, 1, 0};
  ActivationFunctionType acts[] = {RELU, LINEAR};
  ActivationDerivativeType ders[] = {RELU_DERIVATIVE, LINEAR_DERIVATIVE};

  NN_t *nn = NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L2, SGD, 0.01L);
  if (!nn) {
    printf("NN_init failed\n");
    return;
  }

  long double x[] = {1.0L, -2.0L};
  long double *y = NN_forward(nn, x);
  if (y) {
    printf("NN_forward ok, y[0]=%.10Lf\n", y[0]);
    free(y);
  } else {
    printf("NN_forward returned NULL\n");
  }

  NN_destroy(nn);
}

static void smoke_test_neat(void) {
  printf("== NEAT smoke test ==\n");

  // small pop just to ensure it constructs and can run one evolve step
  NEAT_t *neat = NEAT_init(2, 1, 10);
  if (!neat) {
    printf("NEAT_init failed\n");
    return;
  }

  // 2 samples, input dim=2, output dim=1
  long double in0[] = {0.0L, 0.0L};
  long double in1[] = {1.0L, 1.0L};
  long double t0[] = {0.0L};
  long double t1[] = {1.0L};

  long double *inputs[] = {in0, in1};
  long double *targets[] = {t0, t1};

  NEAT_train(neat, inputs, targets, 2);
  printf("NEAT_train ok (evaluated + evolved)\n");

  NEAT_destroy(neat);
}

static void smoke_test_transformer(void) {
  printf("== TRANSFORMER smoke test ==\n");

  size_t D = 8; // model_dim
  size_t H = 2; // heads
  size_t L = 1; // layers
  size_t T = 3; // seq length

  Transformer_t *tr = TRANSFORMER_init(D, H, L);
  if (!tr) {
    printf("TRANSFORMER_init failed\n");
    return;
  }

  // input sequence T x D
  long double **seq = (long double **)malloc(T * sizeof(long double *));
  for (size_t t = 0; t < T; t++) {
    seq[t] = (long double *)calloc(D, sizeof(long double));
    for (size_t i = 0; i < D; i++)
      seq[t][i] = (long double)(t + i) * 0.01L;
  }

  long double target[8] = {0}; // train toward zeros for a single step
  TRANSFORMER_train(tr, seq, T, target);
  printf("TRANSFORMER_train ok (one step)\n");

  // cleanup input seq (train() frees its own forward outputs, not your inputs)
  for (size_t t = 0; t < T; t++)
    free(seq[t]);
  free(seq);

  TRANSFORMER_destroy(tr);
}

int main(void) {
  srand((unsigned int)time(NULL));

  smoke_test_nn();
  smoke_test_neat();
  smoke_test_transformer();

  printf("All smoke tests ran.\n");
  return 0;
}
