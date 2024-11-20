#include "nn_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void serialize_neural_network(NN_t* nn, NeuralNetworkMessage* msg) {
    if (!nn || !msg) return;

    // Serialize network structure
    msg->num_layers = nn->numLayers;
    memcpy(msg->layer_sizes, nn->layers, nn->numLayers * sizeof(size_t));

    // Calculate total nodes and connections
    msg->num_nodes = 0;
    msg->num_connections = 0;
    for (size_t i = 0; i < nn->numLayers; i++) {
        msg->num_nodes += nn->layers[i];
        if (i < nn->numLayers - 1) {
            msg->num_connections += nn->layers[i] * nn->layers[i + 1];
        }
    }

    // Serialize nodes
    int node_idx = 0;
    for (size_t layer = 0; layer < nn->numLayers; layer++) {
        for (size_t node = 0; node < nn->layers[layer]; node++) {
            msg->nodes[node_idx].id = node_idx;
            msg->nodes[node_idx].layer = layer;
            node_idx++;
        }
    }

    // Serialize connections (weights)
    int conn_idx = 0;
    for (size_t layer = 0; layer < nn->numLayers - 1; layer++) {
        for (size_t from = 0; from < nn->layers[layer]; from++) {
            for (size_t to = 0; to < nn->layers[layer + 1]; to++) {
                msg->connections[conn_idx].from_node = from;
                msg->connections[conn_idx].to_node = to;
                msg->connections[conn_idx].weight = nn->weights[layer][from * nn->layers[layer + 1] + to];
                msg->connections[conn_idx].enabled = true;
                conn_idx++;
            }
        }
    }
}

void deserialize_neural_network(NeuralNetworkMessage* msg, NN_t* nn) {
    if (!msg || !nn) return;

    // Initialize network structure
    nn->numLayers = msg->num_layers;
    nn->layers = malloc(nn->numLayers * sizeof(size_t));
    memcpy(nn->layers, msg->layer_sizes, nn->numLayers * sizeof(size_t));

    // Allocate weights
    nn->weights = malloc((nn->numLayers - 1) * sizeof(long double*));
    for (size_t i = 0; i < nn->numLayers - 1; i++) {
        nn->weights[i] = calloc(nn->layers[i] * nn->layers[i + 1], sizeof(long double));
    }

    // Set weights from connections
    for (size_t i = 0; i < msg->num_connections; i++) {
        ConnectionMessage* conn = &msg->connections[i];
        if (conn->enabled) {
            size_t layer = msg->nodes[conn->from_node].layer;
            nn->weights[layer][conn->from_node * nn->layers[layer + 1] + conn->to_node] = conn->weight;
        }
    }
}

int SerializeNN(NN_t* nn, char* buffer, size_t bufsize) {
    if (!nn || !buffer || bufsize < sizeof(NeuralNetworkMessage)) {
        return -1;
    }

    NeuralNetworkMessage msg;
    memset(&msg, 0, sizeof(NeuralNetworkMessage));
    msg.type = MSG_TYPE_NN_DATA;

    serialize_neural_network(nn, &msg);
    memcpy(buffer, &msg, sizeof(NeuralNetworkMessage));
    return sizeof(NeuralNetworkMessage);
}

NN_t* DeserializeNN(const char* buffer, size_t bufsize) {
    if (!buffer || bufsize < sizeof(NeuralNetworkMessage)) {
        return NULL;
    }

    NeuralNetworkMessage msg;
    memcpy(&msg, buffer, sizeof(NeuralNetworkMessage));

    if (msg.type != MSG_TYPE_NN_DATA) {
        return NULL;
    }

    NN_t* nn = malloc(sizeof(NN_t));
    if (!nn) return NULL;

    deserialize_neural_network(&msg, nn);
    return nn;
}
