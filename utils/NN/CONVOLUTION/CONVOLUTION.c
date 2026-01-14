#include "../NN/NN.h"
#include "CONVOLUTION.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ConvLayer {
  ConvLayerType type;
  int in_w, in_h, in_c;
  int out_w, out_h, out_c;
  int k_h, k_w;
  int stride;
  int pad;
  float alpha;
  int depth_mult;
  float dropout_p;
  int residual_from;

  float *dropout_mask;

  float *bn_gamma;
  float *bn_beta;
  float *bn_running_mean;
  float *bn_running_var;
  float *bn_mean;
  float *bn_var;
  float *bn_dgamma;
  float *bn_dbeta;
  float bn_eps;
  float bn_momentum;

  int in_dim;
  int out_dim;

  float *weights;
  float *biases;
  float *dweights;
  float *dbiases;

  float *input;
  float *output;
  int input_size;
  int output_size;
  int *max_idx;
};

struct ConvNet {
  int input_w, input_h, input_c;
  int out_w, out_h, out_c;
  float lr;
  ConvLayer **layers;
  int count;
  int cap;
  float *last_output;
  int last_output_size;
};

static float frand_uniform(void) { return (float)rand() / (float)RAND_MAX; }

static float he_scale(int fan_in) {
  if (fan_in <= 0)
    return 1.0f;
  return sqrtf(2.0f / (float)fan_in);
}

static ConvLayer *layer_create(ConvLayerType type) {
  ConvLayer *l = (ConvLayer *)calloc(1, sizeof(ConvLayer));
  if (!l)
    return NULL;
  l->type = type;
  return l;
}

static void layer_free(ConvLayer *l) {
  if (!l)
    return;
  free(l->weights);
  free(l->biases);
  free(l->dweights);
  free(l->dbiases);
  free(l->dropout_mask);
  free(l->bn_gamma);
  free(l->bn_beta);
  free(l->bn_running_mean);
  free(l->bn_running_var);
  free(l->bn_mean);
  free(l->bn_var);
  free(l->bn_dgamma);
  free(l->bn_dbeta);
  free(l->input);
  free(l->output);
  free(l->max_idx);
  free(l);
}

static int layer_init_buffers(ConvLayer *l) {
  if (!l)
    return 0;
  l->input_size = l->in_w * l->in_h * l->in_c;
  l->output_size = l->out_w * l->out_h * l->out_c;
  l->input = (float *)calloc((size_t)l->input_size, sizeof(float));
  l->output = (float *)calloc((size_t)l->output_size, sizeof(float));
  if (!l->input || !l->output)
    return 0;
  return 1;
}

static int add_layer(ConvNet *net, ConvLayer *layer) {
  if (!net || !layer)
    return 0;
  if (net->count == net->cap) {
    int new_cap = net->cap ? net->cap * 2 : 8;
    ConvLayer **nl =
        (ConvLayer **)realloc(net->layers, sizeof(ConvLayer *) * new_cap);
    if (!nl)
      return 0;
    net->layers = nl;
    net->cap = new_cap;
  }
  net->layers[net->count++] = layer;
  net->out_w = layer->out_w;
  net->out_h = layer->out_h;
  net->out_c = layer->out_c;
  return 1;
}

ConvNet *convnet_create(int input_w, int input_h, int input_c, float lr) {
  ConvNet *net = (ConvNet *)calloc(1, sizeof(ConvNet));
  if (!net)
    return NULL;
  net->input_w = input_w;
  net->input_h = input_h;
  net->input_c = input_c;
  net->out_w = input_w;
  net->out_h = input_h;
  net->out_c = input_c;
  net->lr = (lr > 0.0f) ? lr : 0.001f;
  return net;
}

void convnet_free(ConvNet *net) {
  if (!net)
    return;
  for (int i = 0; i < net->count; i++)
    layer_free(net->layers[i]);
  free(net->layers);
  free(net->last_output);
  free(net);
}

int convnet_add_conv2d(ConvNet *net, int out_c, int k_h, int k_w, int stride,
                       int pad) {
  if (!net || out_c <= 0 || k_h <= 0 || k_w <= 0 || stride <= 0)
    return 0;
  ConvLayer *l = layer_create(LAYER_CONV2D);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_c = out_c;
  l->k_h = k_h;
  l->k_w = k_w;
  l->stride = stride;
  l->pad = pad;
  l->out_w = (l->in_w + 2 * pad - k_w) / stride + 1;
  l->out_h = (l->in_h + 2 * pad - k_h) / stride + 1;
  if (l->out_w <= 0 || l->out_h <= 0) {
    layer_free(l);
    return 0;
  }
  int wcount = out_c * l->in_c * k_h * k_w;
  l->weights = (float *)calloc((size_t)wcount, sizeof(float));
  l->biases = (float *)calloc((size_t)out_c, sizeof(float));
  l->dweights = (float *)calloc((size_t)wcount, sizeof(float));
  l->dbiases = (float *)calloc((size_t)out_c, sizeof(float));
  if (!l->weights || !l->biases || !l->dweights || !l->dbiases ||
      !layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  float scale = he_scale(l->in_c * k_h * k_w);
  for (int i = 0; i < wcount; i++)
    l->weights[i] = (frand_uniform() * 2.0f - 1.0f) * scale;
  return add_layer(net, l);
}

int convnet_add_depthwise_conv2d(ConvNet *net, int depth_mult, int k_h, int k_w,
                                 int stride, int pad) {
  if (!net || depth_mult <= 0 || k_h <= 0 || k_w <= 0 || stride <= 0)
    return 0;
  ConvLayer *l = layer_create(LAYER_DEPTHWISE_CONV2D);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->depth_mult = depth_mult;
  l->out_c = l->in_c * depth_mult;
  l->k_h = k_h;
  l->k_w = k_w;
  l->stride = stride;
  l->pad = pad;
  l->out_w = (l->in_w + 2 * pad - k_w) / stride + 1;
  l->out_h = (l->in_h + 2 * pad - k_h) / stride + 1;
  if (l->out_w <= 0 || l->out_h <= 0) {
    layer_free(l);
    return 0;
  }
  int wcount = l->out_c * k_h * k_w;
  l->weights = (float *)calloc((size_t)wcount, sizeof(float));
  l->biases = (float *)calloc((size_t)l->out_c, sizeof(float));
  l->dweights = (float *)calloc((size_t)wcount, sizeof(float));
  l->dbiases = (float *)calloc((size_t)l->out_c, sizeof(float));
  if (!l->weights || !l->biases || !l->dweights || !l->dbiases ||
      !layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  float scale = he_scale(k_h * k_w);
  for (int i = 0; i < wcount; i++)
    l->weights[i] = (frand_uniform() * 2.0f - 1.0f) * scale;
  return add_layer(net, l);
}

int convnet_add_relu(ConvNet *net) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_RELU);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_leaky_relu(ConvNet *net, float alpha) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_LEAKY_RELU);
  if (!l)
    return 0;
  l->alpha = alpha;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_sigmoid(ConvNet *net) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_SIGMOID);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_tanh(ConvNet *net) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_TANH);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_batchnorm(ConvNet *net, float eps, float momentum) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_BATCHNORM);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  l->bn_eps = (eps > 0.0f) ? eps : 1e-5f;
  l->bn_momentum = (momentum > 0.0f && momentum < 1.0f) ? momentum : 0.9f;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  l->bn_gamma = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_beta = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_running_mean = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_running_var = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_mean = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_var = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_dgamma = (float *)calloc((size_t)l->in_c, sizeof(float));
  l->bn_dbeta = (float *)calloc((size_t)l->in_c, sizeof(float));
  if (!l->bn_gamma || !l->bn_beta || !l->bn_running_mean ||
      !l->bn_running_var || !l->bn_mean || !l->bn_var || !l->bn_dgamma ||
      !l->bn_dbeta) {
    layer_free(l);
    return 0;
  }
  for (int i = 0; i < l->in_c; i++) {
    l->bn_gamma[i] = 1.0f;
    l->bn_beta[i] = 0.0f;
    l->bn_running_mean[i] = 0.0f;
    l->bn_running_var[i] = 1.0f;
  }
  return add_layer(net, l);
}

int convnet_add_maxpool2d(ConvNet *net, int pool_h, int pool_w, int stride) {
  if (!net || pool_h <= 0 || pool_w <= 0 || stride <= 0)
    return 0;
  ConvLayer *l = layer_create(LAYER_MAXPOOL2D);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->k_h = pool_h;
  l->k_w = pool_w;
  l->stride = stride;
  l->out_w = (l->in_w - pool_w) / stride + 1;
  l->out_h = (l->in_h - pool_h) / stride + 1;
  l->out_c = l->in_c;
  if (l->out_w <= 0 || l->out_h <= 0) {
    layer_free(l);
    return 0;
  }
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  l->max_idx = (int *)calloc((size_t)l->output_size, sizeof(int));
  if (!l->max_idx) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_avgpool2d(ConvNet *net, int pool_h, int pool_w, int stride) {
  if (!net || pool_h <= 0 || pool_w <= 0 || stride <= 0)
    return 0;
  ConvLayer *l = layer_create(LAYER_AVGPOOL2D);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->k_h = pool_h;
  l->k_w = pool_w;
  l->stride = stride;
  l->out_w = (l->in_w - pool_w) / stride + 1;
  l->out_h = (l->in_h - pool_h) / stride + 1;
  l->out_c = l->in_c;
  if (l->out_w <= 0 || l->out_h <= 0) {
    layer_free(l);
    return 0;
  }
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_dropout(ConvNet *net, float p) {
  if (!net || p < 0.0f || p >= 1.0f)
    return 0;
  ConvLayer *l = layer_create(LAYER_DROPOUT);
  if (!l)
    return 0;
  l->dropout_p = p;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  l->dropout_mask = (float *)calloc((size_t)l->output_size, sizeof(float));
  if (!l->dropout_mask) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_residual(ConvNet *net, int from_index) {
  if (!net || from_index < 0 || from_index >= net->count)
    return 0;
  ConvLayer *from = net->layers[from_index];
  if (!from)
    return 0;
  ConvLayer *l = layer_create(LAYER_RESIDUAL_ADD);
  if (!l)
    return 0;
  l->residual_from = from_index;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (from->out_w != l->in_w || from->out_h != l->in_h ||
      from->out_c != l->in_c) {
    layer_free(l);
    return 0;
  }
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_flatten(ConvNet *net) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_FLATTEN);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = 1;
  l->out_h = 1;
  l->out_c = l->in_w * l->in_h * l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

int convnet_add_dense(ConvNet *net, int out_dim) {
  if (!net || out_dim <= 0)
    return 0;
  ConvLayer *l = layer_create(LAYER_DENSE);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->in_dim = l->in_w * l->in_h * l->in_c;
  l->out_dim = out_dim;
  l->out_w = 1;
  l->out_h = 1;
  l->out_c = out_dim;
  int wcount = l->out_dim * l->in_dim;
  l->weights = (float *)calloc((size_t)wcount, sizeof(float));
  l->biases = (float *)calloc((size_t)out_dim, sizeof(float));
  l->dweights = (float *)calloc((size_t)wcount, sizeof(float));
  l->dbiases = (float *)calloc((size_t)out_dim, sizeof(float));
  if (!l->weights || !l->biases || !l->dweights || !l->dbiases ||
      !layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  float scale = he_scale(l->in_dim);
  for (int i = 0; i < wcount; i++)
    l->weights[i] = (frand_uniform() * 2.0f - 1.0f) * scale;
  return add_layer(net, l);
}

int convnet_add_softmax(ConvNet *net) {
  if (!net)
    return 0;
  ConvLayer *l = layer_create(LAYER_SOFTMAX);
  if (!l)
    return 0;
  l->in_w = net->out_w;
  l->in_h = net->out_h;
  l->in_c = net->out_c;
  l->out_w = l->in_w;
  l->out_h = l->in_h;
  l->out_c = l->in_c;
  if (!layer_init_buffers(l)) {
    layer_free(l);
    return 0;
  }
  return add_layer(net, l);
}

static void conv2d_forward(ConvLayer *l, const float *in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int out_c = l->out_c;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;
  int pad = l->pad;

  memset(l->output, 0, sizeof(float) * (size_t)l->output_size);
  for (int oc = 0; oc < out_c; oc++) {
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        float sum = l->biases[oc];
        for (int ic = 0; ic < in_c; ic++) {
          for (int ky = 0; ky < k_h; ky++) {
            int iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= in_h)
              continue;
            for (int kx = 0; kx < k_w; kx++) {
              int ix = ox * stride + kx - pad;
              if (ix < 0 || ix >= in_w)
                continue;
              int in_idx = (ic * in_h + iy) * in_w + ix;
              int w_idx = (((oc * in_c + ic) * k_h + ky) * k_w + kx);
              sum += in[in_idx] * l->weights[w_idx];
            }
          }
        }
        int out_idx = (oc * out_h + oy) * out_w + ox;
        l->output[out_idx] = sum;
      }
    }
  }
}

static void conv2d_backward(ConvLayer *l, const float *grad_out,
                            float *grad_in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int out_c = l->out_c;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;
  int pad = l->pad;

  memset(l->dweights, 0,
         sizeof(float) * (size_t)out_c * (size_t)in_c * (size_t)k_h *
             (size_t)k_w);
  memset(l->dbiases, 0, sizeof(float) * (size_t)out_c);
  memset(grad_in, 0, sizeof(float) * (size_t)l->input_size);

  for (int oc = 0; oc < out_c; oc++) {
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        int out_idx = (oc * out_h + oy) * out_w + ox;
        float go = grad_out[out_idx];
        l->dbiases[oc] += go;
        for (int ic = 0; ic < in_c; ic++) {
          for (int ky = 0; ky < k_h; ky++) {
            int iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= in_h)
              continue;
            for (int kx = 0; kx < k_w; kx++) {
              int ix = ox * stride + kx - pad;
              if (ix < 0 || ix >= in_w)
                continue;
              int in_idx = (ic * in_h + iy) * in_w + ix;
              int w_idx = (((oc * in_c + ic) * k_h + ky) * k_w + kx);
              l->dweights[w_idx] += l->input[in_idx] * go;
              grad_in[in_idx] += l->weights[w_idx] * go;
            }
          }
        }
      }
    }
  }
}

static void depthwise_conv2d_forward(ConvLayer *l, const float *in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int out_c = l->out_c;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;
  int pad = l->pad;
  int depth_mult = l->depth_mult;

  memset(l->output, 0, sizeof(float) * (size_t)l->output_size);
  for (int oc = 0; oc < out_c; oc++) {
    int ic = oc / depth_mult;
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        float sum = l->biases[oc];
        for (int ky = 0; ky < k_h; ky++) {
          int iy = oy * stride + ky - pad;
          if (iy < 0 || iy >= in_h)
            continue;
          for (int kx = 0; kx < k_w; kx++) {
            int ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= in_w)
              continue;
            int in_idx = (ic * in_h + iy) * in_w + ix;
            int w_idx = (oc * k_h + ky) * k_w + kx;
            sum += in[in_idx] * l->weights[w_idx];
          }
        }
        int out_idx = (oc * out_h + oy) * out_w + ox;
        l->output[out_idx] = sum;
      }
    }
  }
}

static void depthwise_conv2d_backward(ConvLayer *l, const float *grad_out,
                                      float *grad_in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int out_c = l->out_c;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;
  int pad = l->pad;
  int depth_mult = l->depth_mult;

  memset(l->dweights, 0,
         sizeof(float) * (size_t)out_c * (size_t)k_h * (size_t)k_w);
  memset(l->dbiases, 0, sizeof(float) * (size_t)out_c);
  memset(grad_in, 0, sizeof(float) * (size_t)l->input_size);

  for (int oc = 0; oc < out_c; oc++) {
    int ic = oc / depth_mult;
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        int out_idx = (oc * out_h + oy) * out_w + ox;
        float go = grad_out[out_idx];
        l->dbiases[oc] += go;
        for (int ky = 0; ky < k_h; ky++) {
          int iy = oy * stride + ky - pad;
          if (iy < 0 || iy >= in_h)
            continue;
          for (int kx = 0; kx < k_w; kx++) {
            int ix = ox * stride + kx - pad;
            if (ix < 0 || ix >= in_w)
              continue;
            int in_idx = (ic * in_h + iy) * in_w + ix;
            int w_idx = (oc * k_h + ky) * k_w + kx;
            l->dweights[w_idx] += l->input[in_idx] * go;
            grad_in[in_idx] += l->weights[w_idx] * go;
          }
        }
      }
    }
  }
}

static void relu_forward(ConvLayer *l, const float *in) {
  for (int i = 0; i < l->output_size; i++)
    l->output[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

static void relu_backward(ConvLayer *l, const float *grad_out, float *grad_in) {
  for (int i = 0; i < l->output_size; i++)
    grad_in[i] = (l->input[i] > 0.0f) ? grad_out[i] : 0.0f;
}

static void leaky_relu_forward(ConvLayer *l, const float *in) {
  float a = l->alpha > 0.0f ? l->alpha : 0.01f;
  for (int i = 0; i < l->output_size; i++)
    l->output[i] = (in[i] > 0.0f) ? in[i] : a * in[i];
}

static void leaky_relu_backward(ConvLayer *l, const float *grad_out,
                                float *grad_in) {
  float a = l->alpha > 0.0f ? l->alpha : 0.01f;
  for (int i = 0; i < l->output_size; i++)
    grad_in[i] = (l->input[i] > 0.0f) ? grad_out[i] : a * grad_out[i];
}

static void sigmoid_forward(ConvLayer *l, const float *in) {
  for (int i = 0; i < l->output_size; i++)
    l->output[i] = 1.0f / (1.0f + expf(-in[i]));
}

static void sigmoid_backward(ConvLayer *l, const float *grad_out,
                             float *grad_in) {
  for (int i = 0; i < l->output_size; i++) {
    float y = l->output[i];
    grad_in[i] = grad_out[i] * y * (1.0f - y);
  }
}

static void tanh_forward_layer(ConvLayer *l, const float *in) {
  for (int i = 0; i < l->output_size; i++)
    l->output[i] = tanhf(in[i]);
}

static void tanh_backward_layer(ConvLayer *l, const float *grad_out,
                                float *grad_in) {
  for (int i = 0; i < l->output_size; i++) {
    float y = l->output[i];
    grad_in[i] = grad_out[i] * (1.0f - y * y);
  }
}

static void batchnorm_forward(ConvLayer *l, const float *in) {
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int spatial = in_w * in_h;
  float eps = l->bn_eps;
  float momentum = l->bn_momentum;

  for (int c = 0; c < in_c; c++) {
    float mean = 0.0f;
    for (int i = 0; i < spatial; i++)
      mean += in[c * spatial + i];
    mean /= (float)spatial;

    float var = 0.0f;
    for (int i = 0; i < spatial; i++) {
      float v = in[c * spatial + i] - mean;
      var += v * v;
    }
    var /= (float)spatial;

    l->bn_mean[c] = mean;
    l->bn_var[c] = var;
    l->bn_running_mean[c] =
        momentum * l->bn_running_mean[c] + (1.0f - momentum) * mean;
    l->bn_running_var[c] =
        momentum * l->bn_running_var[c] + (1.0f - momentum) * var;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < spatial; i++) {
      float xhat = (in[c * spatial + i] - mean) * inv_std;
      l->output[c * spatial + i] = l->bn_gamma[c] * xhat + l->bn_beta[c];
    }
  }
}

static void batchnorm_backward(ConvLayer *l, const float *grad_out,
                               float *grad_in) {
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int spatial = in_w * in_h;
  float eps = l->bn_eps;

  memset(l->bn_dgamma, 0, sizeof(float) * (size_t)in_c);
  memset(l->bn_dbeta, 0, sizeof(float) * (size_t)in_c);

  for (int c = 0; c < in_c; c++) {
    float mean = l->bn_mean[c];
    float var = l->bn_var[c];
    float inv_std = 1.0f / sqrtf(var + eps);

    float dgamma = 0.0f;
    float dbeta = 0.0f;
    for (int i = 0; i < spatial; i++) {
      float xhat = (l->input[c * spatial + i] - mean) * inv_std;
      float go = grad_out[c * spatial + i];
      dgamma += go * xhat;
      dbeta += go;
    }
    l->bn_dgamma[c] = dgamma;
    l->bn_dbeta[c] = dbeta;

    for (int i = 0; i < spatial; i++) {
      float xhat = (l->input[c * spatial + i] - mean) * inv_std;
      float go = grad_out[c * spatial + i];
      float term = (float)spatial * go - dbeta - xhat * dgamma;
      grad_in[c * spatial + i] =
          (l->bn_gamma[c] * inv_std / (float)spatial) * term;
    }
  }
}

static void maxpool_forward(ConvLayer *l, const float *in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;

  for (int c = 0; c < in_c; c++) {
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        int in_x = ox * stride;
        int in_y = oy * stride;
        float maxv = -INFINITY;
        int max_i = 0;
        for (int ky = 0; ky < k_h; ky++) {
          int iy = in_y + ky;
          if (iy >= in_h)
            continue;
          for (int kx = 0; kx < k_w; kx++) {
            int ix = in_x + kx;
            if (ix >= in_w)
              continue;
            int in_idx = (c * in_h + iy) * in_w + ix;
            float v = in[in_idx];
            if (v > maxv) {
              maxv = v;
              max_i = in_idx;
            }
          }
        }
        int out_idx = (c * out_h + oy) * out_w + ox;
        l->output[out_idx] = maxv;
        l->max_idx[out_idx] = max_i;
      }
    }
  }
}

static void maxpool_backward(ConvLayer *l, const float *grad_out,
                             float *grad_in) {
  memset(grad_in, 0, sizeof(float) * (size_t)l->input_size);
  for (int i = 0; i < l->output_size; i++) {
    int idx = l->max_idx[i];
    grad_in[idx] += grad_out[i];
  }
}

static void dropout_forward(ConvLayer *l, const float *in) {
  float p = l->dropout_p;
  float scale = (p > 0.0f) ? (1.0f / (1.0f - p)) : 1.0f;
  for (int i = 0; i < l->output_size; i++) {
    float keep = (frand_uniform() >= p) ? 1.0f : 0.0f;
    l->dropout_mask[i] = keep;
    l->output[i] = in[i] * keep * scale;
  }
}

static void dropout_backward(ConvLayer *l, const float *grad_out,
                             float *grad_in) {
  float p = l->dropout_p;
  float scale = (p > 0.0f) ? (1.0f / (1.0f - p)) : 1.0f;
  for (int i = 0; i < l->output_size; i++)
    grad_in[i] = grad_out[i] * l->dropout_mask[i] * scale;
}

static void residual_forward(const ConvNet *net, ConvLayer *l,
                             const float *in) {
  int from = l->residual_from;
  const float *skip = net->layers[from]->output;
  for (int i = 0; i < l->output_size; i++)
    l->output[i] = in[i] + skip[i];
}

static void avgpool_forward(ConvLayer *l, const float *in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;

  float inv = 1.0f / (float)(k_h * k_w);
  for (int c = 0; c < in_c; c++) {
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        int in_x = ox * stride;
        int in_y = oy * stride;
        float sum = 0.0f;
        for (int ky = 0; ky < k_h; ky++) {
          int iy = in_y + ky;
          if (iy >= in_h)
            continue;
          for (int kx = 0; kx < k_w; kx++) {
            int ix = in_x + kx;
            if (ix >= in_w)
              continue;
            int in_idx = (c * in_h + iy) * in_w + ix;
            sum += in[in_idx];
          }
        }
        int out_idx = (c * out_h + oy) * out_w + ox;
        l->output[out_idx] = sum * inv;
      }
    }
  }
}

static void avgpool_backward(ConvLayer *l, const float *grad_out,
                             float *grad_in) {
  int out_w = l->out_w;
  int out_h = l->out_h;
  int in_w = l->in_w;
  int in_h = l->in_h;
  int in_c = l->in_c;
  int k_h = l->k_h;
  int k_w = l->k_w;
  int stride = l->stride;

  memset(grad_in, 0, sizeof(float) * (size_t)l->input_size);
  float inv = 1.0f / (float)(k_h * k_w);
  for (int c = 0; c < in_c; c++) {
    for (int oy = 0; oy < out_h; oy++) {
      for (int ox = 0; ox < out_w; ox++) {
        int out_idx = (c * out_h + oy) * out_w + ox;
        float go = grad_out[out_idx] * inv;
        int in_x = ox * stride;
        int in_y = oy * stride;
        for (int ky = 0; ky < k_h; ky++) {
          int iy = in_y + ky;
          if (iy >= in_h)
            continue;
          for (int kx = 0; kx < k_w; kx++) {
            int ix = in_x + kx;
            if (ix >= in_w)
              continue;
            int in_idx = (c * in_h + iy) * in_w + ix;
            grad_in[in_idx] += go;
          }
        }
      }
    }
  }
}

static void dense_forward(ConvLayer *l, const float *in) {
  for (int o = 0; o < l->out_dim; o++) {
    float sum = l->biases[o];
    for (int i = 0; i < l->in_dim; i++)
      sum += l->weights[o * l->in_dim + i] * in[i];
    l->output[o] = sum;
  }
}

static void dense_backward(ConvLayer *l, const float *grad_out,
                           float *grad_in) {
  memset(l->dweights, 0,
         sizeof(float) * (size_t)l->out_dim * (size_t)l->in_dim);
  memset(l->dbiases, 0, sizeof(float) * (size_t)l->out_dim);
  memset(grad_in, 0, sizeof(float) * (size_t)l->in_dim);
  for (int o = 0; o < l->out_dim; o++) {
    float go = grad_out[o];
    l->dbiases[o] += go;
    for (int i = 0; i < l->in_dim; i++) {
      l->dweights[o * l->in_dim + i] += l->input[i] * go;
      grad_in[i] += l->weights[o * l->in_dim + i] * go;
    }
  }
}

static void softmax_forward(ConvLayer *l, const float *in) {
  float maxv = in[0];
  for (int i = 1; i < l->output_size; i++)
    if (in[i] > maxv)
      maxv = in[i];
  float sum = 0.0f;
  for (int i = 0; i < l->output_size; i++) {
    l->output[i] = expf(in[i] - maxv);
    sum += l->output[i];
  }
  if (sum > 0.0f) {
    float inv = 1.0f / sum;
    for (int i = 0; i < l->output_size; i++)
      l->output[i] *= inv;
  }
}

static void softmax_backward(ConvLayer *l, const float *grad_out,
                             float *grad_in) {
  for (int i = 0; i < l->output_size; i++) {
    float sum = 0.0f;
    for (int j = 0; j < l->output_size; j++) {
      float yi = l->output[i];
      float yj = l->output[j];
      float term = (i == j) ? (yi * (1.0f - yi)) : (-yi * yj);
      sum += term * grad_out[j];
    }
    grad_in[i] = sum;
  }
}

static void add_inplace(float *dst, const float *src, int n) {
  for (int i = 0; i < n; i++)
    dst[i] += src[i];
}

const float *convnet_forward(ConvNet *net, const float *input) {
  if (!net || !input)
    return NULL;

  const float *cur = input;
  for (int i = 0; i < net->count; i++) {
    ConvLayer *l = net->layers[i];
    memcpy(l->input, cur, sizeof(float) * (size_t)l->input_size);

    switch (l->type) {
    case LAYER_CONV2D:
      conv2d_forward(l, cur);
      break;
    case LAYER_DEPTHWISE_CONV2D:
      depthwise_conv2d_forward(l, cur);
      break;
    case LAYER_RELU:
      relu_forward(l, cur);
      break;
    case LAYER_LEAKY_RELU:
      leaky_relu_forward(l, cur);
      break;
    case LAYER_SIGMOID:
      sigmoid_forward(l, cur);
      break;
    case LAYER_TANH:
      tanh_forward_layer(l, cur);
      break;
    case LAYER_BATCHNORM:
      batchnorm_forward(l, cur);
      break;
    case LAYER_MAXPOOL2D:
      maxpool_forward(l, cur);
      break;
    case LAYER_AVGPOOL2D:
      avgpool_forward(l, cur);
      break;
    case LAYER_DROPOUT:
      dropout_forward(l, cur);
      break;
    case LAYER_FLATTEN:
      memcpy(l->output, cur, sizeof(float) * (size_t)l->output_size);
      break;
    case LAYER_DENSE:
      dense_forward(l, cur);
      break;
    case LAYER_SOFTMAX:
      softmax_forward(l, cur);
      break;
    case LAYER_RESIDUAL_ADD:
      residual_forward(net, l, cur);
      break;
    default:
      break;
    }
    cur = l->output;
  }

  int out_size = net->layers[net->count - 1]->output_size;
  if (net->last_output_size != out_size) {
    free(net->last_output);
    net->last_output = (float *)calloc((size_t)out_size, sizeof(float));
    net->last_output_size = out_size;
  }
  if (net->last_output)
    memcpy(net->last_output, cur, sizeof(float) * (size_t)out_size);

  return net->last_output;
}

float *convnet_backward(ConvNet *net, const float *grad_out) {
  if (!net || !grad_out)
    return NULL;

  float *cur_grad = NULL;
  float *next_grad = NULL;
  float **skip_grads = NULL;

  int last = net->count - 1;
  cur_grad =
      (float *)calloc((size_t)net->layers[last]->output_size, sizeof(float));
  if (!cur_grad)
    return NULL;
  memcpy(cur_grad, grad_out,
         sizeof(float) * (size_t)net->layers[last]->output_size);

  skip_grads = (float **)calloc((size_t)net->count, sizeof(float *));
  if (!skip_grads) {
    free(cur_grad);
    return NULL;
  }

  for (int i = last; i >= 0; i--) {
    ConvLayer *l = net->layers[i];
    if (skip_grads[i]) {
      add_inplace(cur_grad, skip_grads[i], l->output_size);
      free(skip_grads[i]);
      skip_grads[i] = NULL;
    }
    next_grad = (float *)calloc((size_t)l->input_size, sizeof(float));
    if (!next_grad) {
      free(cur_grad);
      for (int j = 0; j < net->count; j++)
        free(skip_grads[j]);
      free(skip_grads);
      return NULL;
    }
    switch (l->type) {
    case LAYER_CONV2D:
      conv2d_backward(l, cur_grad, next_grad);
      break;
    case LAYER_DEPTHWISE_CONV2D:
      depthwise_conv2d_backward(l, cur_grad, next_grad);
      break;
    case LAYER_RELU:
      relu_backward(l, cur_grad, next_grad);
      break;
    case LAYER_LEAKY_RELU:
      leaky_relu_backward(l, cur_grad, next_grad);
      break;
    case LAYER_SIGMOID:
      sigmoid_backward(l, cur_grad, next_grad);
      break;
    case LAYER_TANH:
      tanh_backward_layer(l, cur_grad, next_grad);
      break;
    case LAYER_BATCHNORM:
      batchnorm_backward(l, cur_grad, next_grad);
      break;
    case LAYER_MAXPOOL2D:
      maxpool_backward(l, cur_grad, next_grad);
      break;
    case LAYER_AVGPOOL2D:
      avgpool_backward(l, cur_grad, next_grad);
      break;
    case LAYER_DROPOUT:
      dropout_backward(l, cur_grad, next_grad);
      break;
    case LAYER_FLATTEN:
      memcpy(next_grad, cur_grad, sizeof(float) * (size_t)l->input_size);
      break;
    case LAYER_DENSE:
      dense_backward(l, cur_grad, next_grad);
      break;
    case LAYER_SOFTMAX:
      softmax_backward(l, cur_grad, next_grad);
      break;
    case LAYER_RESIDUAL_ADD:
      memcpy(next_grad, cur_grad, sizeof(float) * (size_t)l->input_size);
      if (l->residual_from >= 0 && l->residual_from < net->count) {
        int from = l->residual_from;
        int sz = net->layers[from]->output_size;
        if (!skip_grads[from])
          skip_grads[from] = (float *)calloc((size_t)sz, sizeof(float));
        if (skip_grads[from])
          add_inplace(skip_grads[from], cur_grad, sz);
      }
      break;
    default:
      break;
    }
    free(cur_grad);
    cur_grad = next_grad;
  }

  for (int j = 0; j < net->count; j++)
    free(skip_grads[j]);
  free(skip_grads);

  return cur_grad;
}

void convnet_zero_grad(ConvNet *net) {
  if (!net)
    return;
  for (int i = 0; i < net->count; i++) {
    ConvLayer *l = net->layers[i];
    if (l->type == LAYER_CONV2D) {
      int wcount = l->out_c * l->in_c * l->k_h * l->k_w;
      if (l->dweights)
        memset(l->dweights, 0, sizeof(float) * (size_t)wcount);
      if (l->dbiases)
        memset(l->dbiases, 0, sizeof(float) * (size_t)l->out_c);
    } else if (l->type == LAYER_DEPTHWISE_CONV2D) {
      int wcount = l->out_c * l->k_h * l->k_w;
      if (l->dweights)
        memset(l->dweights, 0, sizeof(float) * (size_t)wcount);
      if (l->dbiases)
        memset(l->dbiases, 0, sizeof(float) * (size_t)l->out_c);
    } else if (l->type == LAYER_DENSE) {
      int wcount = l->out_dim * l->in_dim;
      if (l->dweights)
        memset(l->dweights, 0, sizeof(float) * (size_t)wcount);
      if (l->dbiases)
        memset(l->dbiases, 0, sizeof(float) * (size_t)l->out_dim);
    } else if (l->type == LAYER_BATCHNORM) {
      if (l->bn_dgamma)
        memset(l->bn_dgamma, 0, sizeof(float) * (size_t)l->in_c);
      if (l->bn_dbeta)
        memset(l->bn_dbeta, 0, sizeof(float) * (size_t)l->in_c);
    }
  }
}

void convnet_step(ConvNet *net) {
  if (!net)
    return;
  for (int i = 0; i < net->count; i++) {
    ConvLayer *l = net->layers[i];
    if (l->type == LAYER_CONV2D) {
      int wcount = l->out_c * l->in_c * l->k_h * l->k_w;
      for (int j = 0; j < wcount; j++)
        l->weights[j] -= net->lr * l->dweights[j];
      for (int j = 0; j < l->out_c; j++)
        l->biases[j] -= net->lr * l->dbiases[j];
    } else if (l->type == LAYER_DEPTHWISE_CONV2D) {
      int wcount = l->out_c * l->k_h * l->k_w;
      for (int j = 0; j < wcount; j++)
        l->weights[j] -= net->lr * l->dweights[j];
      for (int j = 0; j < l->out_c; j++)
        l->biases[j] -= net->lr * l->dbiases[j];
    } else if (l->type == LAYER_DENSE) {
      int wcount = l->out_dim * l->in_dim;
      for (int j = 0; j < wcount; j++)
        l->weights[j] -= net->lr * l->dweights[j];
      for (int j = 0; j < l->out_dim; j++)
        l->biases[j] -= net->lr * l->dbiases[j];
    } else if (l->type == LAYER_BATCHNORM) {
      for (int j = 0; j < l->in_c; j++) {
        l->bn_gamma[j] -= net->lr * l->bn_dgamma[j];
        l->bn_beta[j] -= net->lr * l->bn_dbeta[j];
      }
    }
  }
}

int convnet_output_dim(const ConvNet *net) {
  if (!net)
    return 0;
  return net->out_w * net->out_h * net->out_c;
}

int convnet_output_w(const ConvNet *net) { return net ? net->out_w : 0; }
int convnet_output_h(const ConvNet *net) { return net ? net->out_h : 0; }
int convnet_output_c(const ConvNet *net) { return net ? net->out_c : 0; }

static int write_bytes(FILE *f, const void *buf, size_t n) {
  return fwrite(buf, 1, n, f) == n;
}

static int read_bytes(FILE *f, void *buf, size_t n) {
  return fread(buf, 1, n, f) == n;
}

int convnet_save(const ConvNet *net, const char *path) {
  if (!net || !path)
    return 0;
  FILE *f = fopen(path, "wb");
  if (!f)
    return 0;
  const char magic[4] = {'C', 'N', 'V', '1'};
  if (!write_bytes(f, magic, sizeof(magic))) {
    fclose(f);
    return 0;
  }
  if (!write_bytes(f, &net->input_w, sizeof(int)) ||
      !write_bytes(f, &net->input_h, sizeof(int)) ||
      !write_bytes(f, &net->input_c, sizeof(int)) ||
      !write_bytes(f, &net->lr, sizeof(float)) ||
      !write_bytes(f, &net->count, sizeof(int))) {
    fclose(f);
    return 0;
  }
  for (int i = 0; i < net->count; i++) {
    ConvLayer *l = net->layers[i];
    if (!write_bytes(f, &l->type, sizeof(int)) ||
        !write_bytes(f, &l->in_w, sizeof(int)) ||
        !write_bytes(f, &l->in_h, sizeof(int)) ||
        !write_bytes(f, &l->in_c, sizeof(int)) ||
        !write_bytes(f, &l->out_w, sizeof(int)) ||
        !write_bytes(f, &l->out_h, sizeof(int)) ||
        !write_bytes(f, &l->out_c, sizeof(int)) ||
        !write_bytes(f, &l->k_h, sizeof(int)) ||
        !write_bytes(f, &l->k_w, sizeof(int)) ||
        !write_bytes(f, &l->stride, sizeof(int)) ||
        !write_bytes(f, &l->pad, sizeof(int)) ||
        !write_bytes(f, &l->alpha, sizeof(float)) ||
        !write_bytes(f, &l->in_dim, sizeof(int)) ||
        !write_bytes(f, &l->out_dim, sizeof(int)) ||
        !write_bytes(f, &l->depth_mult, sizeof(int)) ||
        !write_bytes(f, &l->dropout_p, sizeof(float)) ||
        !write_bytes(f, &l->residual_from, sizeof(int)) ||
        !write_bytes(f, &l->bn_eps, sizeof(float)) ||
        !write_bytes(f, &l->bn_momentum, sizeof(float))) {
      fclose(f);
      return 0;
    }

    if (l->type == LAYER_CONV2D) {
      int wcount = l->out_c * l->in_c * l->k_h * l->k_w;
      if (!write_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !write_bytes(f, l->biases, sizeof(float) * (size_t)l->out_c)) {
        fclose(f);
        return 0;
      }
    } else if (l->type == LAYER_DEPTHWISE_CONV2D) {
      int wcount = l->out_c * l->k_h * l->k_w;
      if (!write_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !write_bytes(f, l->biases, sizeof(float) * (size_t)l->out_c)) {
        fclose(f);
        return 0;
      }
    } else if (l->type == LAYER_DENSE) {
      int wcount = l->out_dim * l->in_dim;
      if (!write_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !write_bytes(f, l->biases, sizeof(float) * (size_t)l->out_dim)) {
        fclose(f);
        return 0;
      }
    } else if (l->type == LAYER_BATCHNORM) {
      if (!write_bytes(f, l->bn_gamma, sizeof(float) * (size_t)l->in_c) ||
          !write_bytes(f, l->bn_beta, sizeof(float) * (size_t)l->in_c) ||
          !write_bytes(f, l->bn_running_mean,
                       sizeof(float) * (size_t)l->in_c) ||
          !write_bytes(f, l->bn_running_var, sizeof(float) * (size_t)l->in_c)) {
        fclose(f);
        return 0;
      }
    }
  }
  fclose(f);
  return 1;
}

ConvNet *convnet_load(const char *path) {
  if (!path)
    return NULL;
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  char magic[4];
  if (!read_bytes(f, magic, sizeof(magic)) || memcmp(magic, "CNV1", 4) != 0) {
    fclose(f);
    return NULL;
  }
  int input_w = 0, input_h = 0, input_c = 0, count = 0;
  float lr = 0.0f;
  if (!read_bytes(f, &input_w, sizeof(int)) ||
      !read_bytes(f, &input_h, sizeof(int)) ||
      !read_bytes(f, &input_c, sizeof(int)) ||
      !read_bytes(f, &lr, sizeof(float)) ||
      !read_bytes(f, &count, sizeof(int))) {
    fclose(f);
    return NULL;
  }
  ConvNet *net = convnet_create(input_w, input_h, input_c, lr);
  if (!net) {
    fclose(f);
    return NULL;
  }
  for (int i = 0; i < count; i++) {
    ConvLayerType type = LAYER_CONV2D;
    int in_w = 0, in_h = 0, in_c = 0, out_w = 0, out_h = 0, out_c = 0;
    int k_h = 0, k_w = 0, stride = 0, pad = 0;
    float alpha = 0.0f;
    int in_dim = 0, out_dim = 0;
    int depth_mult = 0;
    float dropout_p = 0.0f;
    int residual_from = -1;
    float bn_eps = 0.0f, bn_momentum = 0.0f;

    if (!read_bytes(f, &type, sizeof(int)) ||
        !read_bytes(f, &in_w, sizeof(int)) ||
        !read_bytes(f, &in_h, sizeof(int)) ||
        !read_bytes(f, &in_c, sizeof(int)) ||
        !read_bytes(f, &out_w, sizeof(int)) ||
        !read_bytes(f, &out_h, sizeof(int)) ||
        !read_bytes(f, &out_c, sizeof(int)) ||
        !read_bytes(f, &k_h, sizeof(int)) ||
        !read_bytes(f, &k_w, sizeof(int)) ||
        !read_bytes(f, &stride, sizeof(int)) ||
        !read_bytes(f, &pad, sizeof(int)) ||
        !read_bytes(f, &alpha, sizeof(float)) ||
        !read_bytes(f, &in_dim, sizeof(int)) ||
        !read_bytes(f, &out_dim, sizeof(int)) ||
        !read_bytes(f, &depth_mult, sizeof(int)) ||
        !read_bytes(f, &dropout_p, sizeof(float)) ||
        !read_bytes(f, &residual_from, sizeof(int)) ||
        !read_bytes(f, &bn_eps, sizeof(float)) ||
        !read_bytes(f, &bn_momentum, sizeof(float))) {
      convnet_free(net);
      fclose(f);
      return NULL;
    }

    int ok = 0;
    switch (type) {
    case LAYER_CONV2D:
      ok = convnet_add_conv2d(net, out_c, k_h, k_w, stride, pad);
      break;
    case LAYER_DEPTHWISE_CONV2D:
      ok = convnet_add_depthwise_conv2d(net, depth_mult, k_h, k_w, stride, pad);
      break;
    case LAYER_RELU:
      ok = convnet_add_relu(net);
      break;
    case LAYER_LEAKY_RELU:
      ok = convnet_add_leaky_relu(net, alpha);
      break;
    case LAYER_SIGMOID:
      ok = convnet_add_sigmoid(net);
      break;
    case LAYER_TANH:
      ok = convnet_add_tanh(net);
      break;
    case LAYER_BATCHNORM:
      ok = convnet_add_batchnorm(net, bn_eps, bn_momentum);
      break;
    case LAYER_MAXPOOL2D:
      ok = convnet_add_maxpool2d(net, k_h, k_w, stride);
      break;
    case LAYER_AVGPOOL2D:
      ok = convnet_add_avgpool2d(net, k_h, k_w, stride);
      break;
    case LAYER_DROPOUT:
      ok = convnet_add_dropout(net, dropout_p);
      break;
    case LAYER_FLATTEN:
      ok = convnet_add_flatten(net);
      break;
    case LAYER_DENSE:
      ok = convnet_add_dense(net, out_dim);
      break;
    case LAYER_SOFTMAX:
      ok = convnet_add_softmax(net);
      break;
    case LAYER_RESIDUAL_ADD:
      ok = convnet_add_residual(net, residual_from);
      break;
    default:
      ok = 0;
      break;
    }

    if (!ok) {
      convnet_free(net);
      fclose(f);
      return NULL;
    }

    ConvLayer *l = net->layers[net->count - 1];
    if (l->in_w != in_w || l->in_h != in_h || l->in_c != in_c ||
        l->out_w != out_w || l->out_h != out_h || l->out_c != out_c) {
      convnet_free(net);
      fclose(f);
      return NULL;
    }
    if (type == LAYER_DENSE && (l->in_dim != in_dim || l->out_dim != out_dim)) {
      convnet_free(net);
      fclose(f);
      return NULL;
    }

    if (type == LAYER_CONV2D) {
      int wcount = l->out_c * l->in_c * l->k_h * l->k_w;
      if (!read_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !read_bytes(f, l->biases, sizeof(float) * (size_t)l->out_c)) {
        convnet_free(net);
        fclose(f);
        return NULL;
      }
    } else if (type == LAYER_DEPTHWISE_CONV2D) {
      int wcount = l->out_c * l->k_h * l->k_w;
      if (!read_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !read_bytes(f, l->biases, sizeof(float) * (size_t)l->out_c)) {
        convnet_free(net);
        fclose(f);
        return NULL;
      }
    } else if (type == LAYER_DENSE) {
      int wcount = l->out_dim * l->in_dim;
      if (!read_bytes(f, l->weights, sizeof(float) * (size_t)wcount) ||
          !read_bytes(f, l->biases, sizeof(float) * (size_t)l->out_dim)) {
        convnet_free(net);
        fclose(f);
        return NULL;
      }
    } else if (type == LAYER_BATCHNORM) {
      if (!read_bytes(f, l->bn_gamma, sizeof(float) * (size_t)l->in_c) ||
          !read_bytes(f, l->bn_beta, sizeof(float) * (size_t)l->in_c) ||
          !read_bytes(f, l->bn_running_mean, sizeof(float) * (size_t)l->in_c) ||
          !read_bytes(f, l->bn_running_var, sizeof(float) * (size_t)l->in_c)) {
        convnet_free(net);
        fclose(f);
        return NULL;
      }
    }
  }

  fclose(f);
  return net;
}

CONVNet *CONV_create(int input_w, int input_h, int input_c, float lr) {
  return convnet_create(input_w, input_h, input_c, lr);
}

void CONV_free(CONVNet *net) { convnet_free(net); }

int CONV_add_conv2d(CONVNet *net, int out_c, int k_h, int k_w, int stride,
                    int pad) {
  return convnet_add_conv2d(net, out_c, k_h, k_w, stride, pad);
}

int CONV_add_depthwise_conv2d(CONVNet *net, int depth_mult, int k_h, int k_w,
                              int stride, int pad) {
  return convnet_add_depthwise_conv2d(net, depth_mult, k_h, k_w, stride, pad);
}

int CONV_add_relu(CONVNet *net) { return convnet_add_relu(net); }

int CONV_add_leaky_relu(CONVNet *net, float alpha) {
  return convnet_add_leaky_relu(net, alpha);
}

int CONV_add_sigmoid(CONVNet *net) { return convnet_add_sigmoid(net); }

int CONV_add_tanh(CONVNet *net) { return convnet_add_tanh(net); }

int CONV_add_batchnorm(CONVNet *net, float eps, float momentum) {
  return convnet_add_batchnorm(net, eps, momentum);
}

int CONV_add_maxpool2d(CONVNet *net, int pool_h, int pool_w, int stride) {
  return convnet_add_maxpool2d(net, pool_h, pool_w, stride);
}

int CONV_add_avgpool2d(CONVNet *net, int pool_h, int pool_w, int stride) {
  return convnet_add_avgpool2d(net, pool_h, pool_w, stride);
}

int CONV_add_dropout(CONVNet *net, float p) {
  return convnet_add_dropout(net, p);
}

int CONV_add_residual(CONVNet *net, int from_index) {
  return convnet_add_residual(net, from_index);
}

int CONV_add_flatten(CONVNet *net) { return convnet_add_flatten(net); }

int CONV_add_dense(CONVNet *net, int out_dim) {
  return convnet_add_dense(net, out_dim);
}

int CONV_add_softmax(CONVNet *net) { return convnet_add_softmax(net); }

const float *CONV_forward(CONVNet *net, const float *input) {
  return convnet_forward(net, input);
}

float *CONV_backward(CONVNet *net, const float *grad_out) {
  return convnet_backward(net, grad_out);
}

void CONV_zero_grad(CONVNet *net) { convnet_zero_grad(net); }

void CONV_step(CONVNet *net) { convnet_step(net); }

int CONV_output_dim(const CONVNet *net) { return convnet_output_dim(net); }

int CONV_output_w(const CONVNet *net) { return convnet_output_w(net); }

int CONV_output_h(const CONVNet *net) { return convnet_output_h(net); }

int CONV_output_c(const CONVNet *net) { return convnet_output_c(net); }

int CONV_save(const CONVNet *net, const char *path) {
  return convnet_save(net, path);
}

CONVNet *CONV_load(const char *path) { return convnet_load(path); }
