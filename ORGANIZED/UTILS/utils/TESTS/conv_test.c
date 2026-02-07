#include "../NN/CONV.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frand01(void) { return (float)rand() / (float)RAND_MAX; }

int main(void) {
  srand(1234);

  int w = 8;
  int h = 8;
  int c = 3;

  CONVNet *net = CONV_create(w, h, c, 0.001f);
  if (!net) {
    printf("convnet_create failed\n");
    return 1;
  }

  if (!CONV_add_conv2d(net, 4, 3, 3, 1, 1) || !CONV_add_relu(net) ||
      !CONV_add_maxpool2d(net, 2, 2, 2) ||
      !CONV_add_conv2d(net, 8, 3, 3, 1, 1) || !CONV_add_relu(net) ||
      !CONV_add_flatten(net) || !CONV_add_dense(net, 5) ||
      !CONV_add_softmax(net)) {
    printf("convnet_add_* failed\n");
    CONV_free(net);
    return 1;
  }

  int in_dim = w * h * c;
  float *input = (float *)malloc(sizeof(float) * (size_t)in_dim);
  if (!input) {
    CONV_free(net);
    return 1;
  }
  for (int i = 0; i < in_dim; i++)
    input[i] = frand01();

  const float *out = CONV_forward(net, input);
  if (!out) {
    printf("convnet_forward failed\n");
    free(input);
    CONV_free(net);
    return 1;
  }

  int out_dim = CONV_output_dim(net);
  if (out_dim <= 0) {
    printf("convnet_output_dim failed\n");
    free(input);
    CONV_free(net);
    return 1;
  }

  float *grad = (float *)calloc((size_t)out_dim, sizeof(float));
  if (!grad) {
    free(input);
    CONV_free(net);
    return 1;
  }

  for (int i = 0; i < out_dim; i++)
    grad[i] = (i == 0) ? -1.0f : 0.0f;

  float *gin = CONV_backward(net, grad);
  if (!gin) {
    printf("convnet_backward failed\n");
    free(grad);
    free(input);
    CONV_free(net);
    return 1;
  }

  CONV_step(net);

  printf("conv_test ok out_dim=%d\n", out_dim);

  free(gin);
  free(grad);
  free(input);
  CONV_free(net);
  return 0;
}
