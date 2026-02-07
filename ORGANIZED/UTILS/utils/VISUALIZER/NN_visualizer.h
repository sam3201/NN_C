#ifndef NN_VISUALIZER_H
#define NN_VISUALIZER_H

#include <stdbool.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "../Raylib/src/raylib.h"
#include "../NN/NN.h"

// Window dimensions
#define VIS_WINDOW_WIDTH 800
#define VIS_WINDOW_HEIGHT 600

// Node visualization
#define NODE_RADIUS 10
#define NODE_OUTLINE_THICKNESS 2
#define INPUT_NODE_COLOR BLUE
#define HIDDEN_NODE_COLOR GRAY
#define OUTPUT_NODE_COLOR GREEN
#define NODE_OUTLINE_COLOR BLACK

// Connection visualization
#define CONNECTION_THICKNESS 2
#define MAX_CONNECTION_THICKNESS 5
#define MIN_CONNECTION_ALPHA 0.2f
#define MAX_CONNECTION_ALPHA 1.0f

// Layout
#define LAYER_SPACING 150
#define NODE_VERTICAL_SPACING 50
#define MARGIN 50

// Text
#define FONT_SIZE 12
#define TEXT_PADDING 5

// Function declarations
void InitializeVisualizer(void);
void UpdateVisualizer(void);
void DrawNeuralNetwork(NN_t* nn);
void CloseVisualizer(void);

#endif // NN_VISUALIZER_H
