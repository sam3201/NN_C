#ifndef VISUALIZER_CLIENT_H
#define VISUALIZER_CLIENT_H

#include "../NN/NN.h"

// Initialize connection to visualization server
int InitializeVisualizerClient(void);

// Send neural network data to visualization server
void SendNetworkToVisualizer(NN_t* nn);

// Close connection to visualization server
void CloseVisualizerClient(void);

#endif // VISUALIZER_CLIENT_H
