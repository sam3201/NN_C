#include "NN_visualizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static bool visualizerInitialized = false;

void InitializeVisualizer(void) {
    if (!visualizerInitialized) {
        InitWindow(VIS_WINDOW_WIDTH, VIS_WINDOW_HEIGHT, "Neural Network Visualizer");
        SetTargetFPS(60);
        visualizerInitialized = true;
    }
}

void UpdateVisualizer(void) {
    if (!visualizerInitialized) return;
    
    if (WindowShouldClose()) {
        CloseVisualizer();
    }
}

void DrawNeuralNetwork(NN_t* nn) {
    if (!visualizerInitialized || !nn) return;

    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Calculate dimensions
    int maxNodesPerLayer = 0;
    for (int i = 0; i < nn->numLayers; i++) {
        if (nn->layers[i] > maxNodesPerLayer) {
            maxNodesPerLayer = nn->layers[i];
        }
    }

    float totalHeight = (maxNodesPerLayer - 1) * NODE_VERTICAL_SPACING;
    float startY = (VIS_WINDOW_HEIGHT - totalHeight) / 2;

    // Draw connections first (so they appear behind nodes)
    for (int layer = 0; layer < nn->numLayers - 1; layer++) {
        for (int i = 0; i < nn->layers[layer]; i++) {
            float startX = MARGIN + (layer * LAYER_SPACING);
            float fromY = startY + (i * NODE_VERTICAL_SPACING);

            for (int j = 0; j < nn->layers[layer + 1]; j++) {
                float endX = MARGIN + ((layer + 1) * LAYER_SPACING);
                float toY = startY + (j * NODE_VERTICAL_SPACING);

                // Get weight and calculate connection properties
                float weight = (float)nn->weights[layer][i * nn->layers[layer + 1] + j];
                float alpha = (fabsf(weight) * (MAX_CONNECTION_ALPHA - MIN_CONNECTION_ALPHA)) + MIN_CONNECTION_ALPHA;
                if (alpha > MAX_CONNECTION_ALPHA) alpha = MAX_CONNECTION_ALPHA;
                
                Color lineColor = {0, 0, 0, (unsigned char)(alpha * 255)};
                float thickness = fabsf(weight) * MAX_CONNECTION_THICKNESS;
                if (thickness < CONNECTION_THICKNESS) thickness = CONNECTION_THICKNESS;
                if (thickness > MAX_CONNECTION_THICKNESS) thickness = MAX_CONNECTION_THICKNESS;

                DrawLineEx((Vector2){startX, fromY}, (Vector2){endX, toY}, thickness, lineColor);
            }
        }
    }

    // Draw nodes
    for (int layer = 0; layer < nn->numLayers; layer++) {
        float x = MARGIN + (layer * LAYER_SPACING);

        for (int node = 0; node < nn->layers[layer]; node++) {
            float y = startY + (node * NODE_VERTICAL_SPACING);

            // Choose node color based on layer
            Color nodeColor;
            if (layer == 0) nodeColor = INPUT_NODE_COLOR;
            else if (layer == nn->numLayers - 1) nodeColor = OUTPUT_NODE_COLOR;
            else nodeColor = HIDDEN_NODE_COLOR;

            // Draw node
            DrawCircle(x, y, NODE_RADIUS, nodeColor);
            DrawCircleLines(x, y, NODE_RADIUS, NODE_OUTLINE_COLOR);
        }
    }

    EndDrawing();
}

void CloseVisualizer(void) {
    if (visualizerInitialized) {
        CloseWindow();
        visualizerInitialized = false;
    }
}
