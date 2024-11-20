#ifndef NN_PROTOCOL_H
#define NN_PROTOCOL_H

#include <stdbool.h>
#include "../NN/NN.h"

#define MAX_LAYERS 10
#define MAX_NODES 1000
#define MAX_CONNECTIONS 5000
#define MSG_TYPE_NN_DATA 1

typedef struct {
    int id;
    int layer;
} NodeMessage;

typedef struct {
    int from_node;
    int to_node;
    long double weight;
    bool enabled;
} ConnectionMessage;

typedef struct {
    int type;
    size_t num_layers;
    size_t layer_sizes[MAX_LAYERS];
    size_t num_nodes;
    size_t num_connections;
    NodeMessage nodes[MAX_NODES];
    ConnectionMessage connections[MAX_CONNECTIONS];
} NeuralNetworkMessage;

void serialize_neural_network(NN_t* nn, NeuralNetworkMessage* msg);
void deserialize_neural_network(NeuralNetworkMessage* msg, NN_t* nn);
int SerializeNN(NN_t* nn, char* buffer, size_t bufsize);
NN_t* DeserializeNN(const char* buffer, size_t bufsize);

#endif // NN_PROTOCOL_H
