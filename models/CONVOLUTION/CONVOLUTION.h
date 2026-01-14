#ifndef NN_CONV_H
#define NN_CONV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LAYER_CONV2D = 0,
  LAYER_DEPTHWISE_CONV2D,
  LAYER_RELU,
  LAYER_LEAKY_RELU,
  LAYER_SIGMOID,
  LAYER_TANH,
  LAYER_BATCHNORM,
  LAYER_MAXPOOL2D,
  LAYER_AVGPOOL2D,
  LAYER_DROPOUT,
  LAYER_FLATTEN,
  LAYER_DENSE,
  LAYER_SOFTMAX,
  LAYER_RESIDUAL_ADD
} ConvLayerType;

typedef struct ConvLayer ConvLayer;
typedef struct ConvNet ConvNet;

ConvNet *convnet_create(int input_w, int input_h, int input_c, float lr);
void convnet_free(ConvNet *net);

int convnet_add_conv2d(ConvNet *net, int out_c, int k_h, int k_w, int stride,
                       int pad);
int convnet_add_depthwise_conv2d(ConvNet *net, int depth_mult, int k_h, int k_w,
                                 int stride, int pad);
int convnet_add_relu(ConvNet *net);
int convnet_add_leaky_relu(ConvNet *net, float alpha);
int convnet_add_sigmoid(ConvNet *net);
int convnet_add_tanh(ConvNet *net);
int convnet_add_batchnorm(ConvNet *net, float eps, float momentum);
int convnet_add_maxpool2d(ConvNet *net, int pool_h, int pool_w, int stride);
int convnet_add_avgpool2d(ConvNet *net, int pool_h, int pool_w, int stride);
int convnet_add_dropout(ConvNet *net, float p);
int convnet_add_residual(ConvNet *net, int from_index);
int convnet_add_flatten(ConvNet *net);
int convnet_add_dense(ConvNet *net, int out_dim);
int convnet_add_softmax(ConvNet *net);

/* Forward returns pointer to internal output buffer; valid until next forward. */
const float *convnet_forward(ConvNet *net, const float *input);

/* Backward takes grad_out of last layer output and returns newly allocated
   grad_input (caller must free). */
float *convnet_backward(ConvNet *net, const float *grad_out);

void convnet_zero_grad(ConvNet *net);
void convnet_step(ConvNet *net);

int convnet_output_dim(const ConvNet *net);
int convnet_output_w(const ConvNet *net);
int convnet_output_h(const ConvNet *net);
int convnet_output_c(const ConvNet *net);

int convnet_save(const ConvNet *net, const char *path);
ConvNet *convnet_load(const char *path);

/* Uppercase API */
typedef ConvNet CONVNet;
CONVNet *CONV_create(int input_w, int input_h, int input_c, float lr);
void CONV_free(CONVNet *net);
int CONV_add_conv2d(CONVNet *net, int out_c, int k_h, int k_w, int stride,
                    int pad);
int CONV_add_depthwise_conv2d(CONVNet *net, int depth_mult, int k_h, int k_w,
                              int stride, int pad);
int CONV_add_relu(CONVNet *net);
int CONV_add_leaky_relu(CONVNet *net, float alpha);
int CONV_add_sigmoid(CONVNet *net);
int CONV_add_tanh(CONVNet *net);
int CONV_add_batchnorm(CONVNet *net, float eps, float momentum);
int CONV_add_maxpool2d(CONVNet *net, int pool_h, int pool_w, int stride);
int CONV_add_avgpool2d(CONVNet *net, int pool_h, int pool_w, int stride);
int CONV_add_dropout(CONVNet *net, float p);
int CONV_add_residual(CONVNet *net, int from_index);
int CONV_add_flatten(CONVNet *net);
int CONV_add_dense(CONVNet *net, int out_dim);
int CONV_add_softmax(CONVNet *net);
const float *CONV_forward(CONVNet *net, const float *input);
float *CONV_backward(CONVNet *net, const float *grad_out);
void CONV_zero_grad(CONVNet *net);
void CONV_step(CONVNet *net);
int CONV_output_dim(const CONVNet *net);
int CONV_output_w(const CONVNet *net);
int CONV_output_h(const CONVNet *net);
int CONV_output_c(const CONVNet *net);
int CONV_save(const CONVNet *net, const char *path);
CONVNet *CONV_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif
