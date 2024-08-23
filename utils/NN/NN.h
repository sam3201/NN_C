#ifndef NN_H
#define NN_H

#include <math.h>
#include <stddef.h>

typedef struct NN_t {
    size_t *layers;
    size_t numLayers;

    long double **weights;
    long double **biases;

    long double (**activationFunctions)(long double);
    long double (**activationDerivatives)(long double);

    long double (*lossFunction)(long double, long double);
    long double (*lossDerivative)(long double, long double);

} NN_t;

NN_t *NN_init(size_t *layers, size_t numLayers,
              long double (**activationFunctions)(long double), long double (**activationDerivatives)(long double),
              long double (*lossFunction)(long double, long double), long double (*lossDerivative)(long double, long double));

void NN_add_layer(NN_t *nn, size_t layerSize, long double (**activationFunctions)(long double), long double (**activationDerivatives)(long double));

long double NN_matmul(long double *inputs, long double *weights, long double *biases); 
long double *NN_forward(NN_t *nn, long double *inputs);
void NN_backprop(NN_t *nn, long double *inputs, long double *outputs, long double *labels);

void NN_destroy(NN_t *nn);

long double sigmoid(long double x);
long double sigmoid_derivative(long double x);

long double relu(long double x);
long double relu_derivative(long double x);

long double tanh_activation(long double x);
long double tanh_derivative(long double x);

long double argmax(long double x);
long double argmax_derivative(long double x);

long double softmax(long double *x);
long double softmax_derivative(long double *x);

long double mse(long double y_true, long double y_pred);
long double mse_derivative(long double y_true, long double y_pred);

#endif //NN_H

