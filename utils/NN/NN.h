#ifndef NN_H
#define NN_H

#include <stddef.h>
#include <math.h>

typedef long double (*ActivationFunction)(long double x);
typedef long double (*ActivationDerivative)(long double x);
typedef long double (*LossFunction)(long double y_true, long double y_pred);
typedef long double (*LossDerivative)(long double y_true, long double y_pred);

typedef enum {
    SIGMOID,
    RELU,
    TANH,
    ARGMAX,
    SOFTMAX,
    ACTIVATION_FUNCTION_COUNT
} ActivationFunctionType;

typedef enum {
    SIGMOID_DERIVATIVE,
    RELU_DERIVATIVE,
    TANH_DERIVATIVE,
    ARGMAX_DERIVATIVE,
    SOFTMAX_DERIVATIVE,
    ACTIVATION_DERIVATIVE_COUNT 
} ActivationDerivativeType;

typedef enum {
   MSE,
   CE,
   MAE,
   LOSS_FUNCTION_COUNT
} LossFunctionType;

typedef enum {
   MSE_DERIVATIVE,
   CE_DERIVATIVE,
   MAE_DERIVATIVE,
   LOSS_DERIVATIVE_COUNT
} LossDerivativeType;

typedef struct {
    size_t *layers;
    size_t numLayers;
        
    long double **weights;
    long double **biases;

    ActivationFunction *activationFunctions; 
    ActivationFunction *activationDerivatives;  

    LossFunction lossFunction; 
    LossDerivative lossDerivative; 
    
    long double learningRate;
} NN_t;

NN_t *NN_init(size_t layers[],
              ActivationFunctionType activationFunctions[], ActivationDerivativeType activationDerivatives[],
              LossFunctionType lossFunction, LossDerivativeType lossDerivative, long double learningRate);

NN_t *NN_init_random(unsigned int num_inputs, unsigned int num_outputs);

// Matrix multiplication function
long double *NN_matmul(long double inputs[], long double weights[], long double biases[], size_t input_size, size_t output_size);

long double *NN_forward(NN_t *nn, long double inputs[]);
long double NN_loss(NN_t *nn, long double y_true, long double y_predicted); 
long double NN_loss_derivative(NN_t *nn, long double y_true, long double y_predicted);
void NN_backprop(NN_t *nn, long double inputs[], long double y_true, long double y_predicted); 
void NN_train(NN_t *nn, long double *inputs, long double *targets, size_t num_targets);

void NN_destroy(NN_t *nn);

NN_t *NN_copy(NN_t *nn);
void NN_mutate(NN_t *nn, long double mutationRate, long double mutationStrength); 
NN_t *NN_crossover(NN_t *parent1, NN_t *parent2);
void NN_rl_backprop(NN_t *nn, long double *inputs, long double *y_true, 
                    long double *y_predicted, long double *rewards, 
                    long double gamma); 

long double sigmoid(long double x);
long double sigmoid_derivative(long double x);
long double relu(long double x);
long double relu_derivative(long double x);
long double tanh_activation(long double x);
long double tanh_derivative(long double x);
long double argmax(long double x[]);
long double argmax_derivative(long double x[]);
long double softmax(long double x);
long double softmax_derivative(long double x);

long double mse(long double y_true, long double y_pred);
long double mse_derivative(long double y_true, long double y_pred);
long double mae(long double y_true, long double y_pred);
long double mae_derivative(long double y_true, long double y_pred);
long double ce(long double y_true, long double y_pred);
long double ce_derivative(long double y_true, long double y_pred);

// Function type to enum conversion
ActivationFunctionType activation_function_to_enum(ActivationFunction func);
ActivationDerivativeType activation_derivative_to_enum(ActivationDerivative func);

// Save and load functions
int NN_save(NN_t *nn, const char *filename);
NN_t *NN_load(const char *filename);

#endif // NN_H
