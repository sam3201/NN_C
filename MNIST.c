#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/Raylib/raylib.h"
#include "utils/NN/NN.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define DRAWING_AREA_SIZE 300 
#define GRID_SIZE 30 
#define CELL_SIZE (DRAWING_AREA_SIZE / GRID_SIZE)

#define NUM_LAYERS 3 
#define HIDDEN_SIZE 4 
#define OUTPUT_SIZE 10

void InitializeNN(NN_t **nn);
void Draw_Grid(int startX, int startY);
void UpdateDrawing(int startX, int startY, long double *input);
int RecognizeDigit(NN_t *nn, long double *input);

int main(void) {
    SetTraceLogLevel(LOG_ERROR); 
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "MNIST Digit Recognition");
    SetTargetFPS(60);

    NN_t *nn;
    InitializeNN(&nn);

    long double input[GRID_SIZE * GRID_SIZE] = {0};
    int startX = (SCREEN_WIDTH - DRAWING_AREA_SIZE) / 2;
    int startY = (SCREEN_HEIGHT - DRAWING_AREA_SIZE) / 2;

    int recognizedDigit = -1;
    bool isCorrect = false;
    bool isDigitRecognized = false;

    Rectangle correctButton = {10, SCREEN_HEIGHT - 80, 150, 30};
    Rectangle incorrectButton = {170, SCREEN_HEIGHT - 80, 150, 30};

    while (!WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            UpdateDrawing(startX, startY, input);
        }

        if (IsKeyPressed(KEY_SPACE)) {
            recognizedDigit = RecognizeDigit(nn, input);
            isDigitRecognized = true;
        }

        if (IsKeyPressed(KEY_C)) {
            memset(input, 0, sizeof(input));
            recognizedDigit = -1;
            isDigitRecognized = false;
            isCorrect = false;
        }

        if (isDigitRecognized) {
            Vector2 mousePoint = GetMousePosition();
            if (CheckCollisionPointRec(mousePoint, correctButton) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                isCorrect = true;
                memset(input, 0, sizeof(input)); // Clear the drawing
                recognizedDigit = -1;
                isDigitRecognized = false;
            } else if (CheckCollisionPointRec(mousePoint, incorrectButton) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                isCorrect = false;
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);
        Draw_Grid(startX, startY);
        
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                int index = i * GRID_SIZE + j;
                Color cellColor = ColorAlpha(BLACK, (float)input[index]);
                DrawRectangle(startX + j * CELL_SIZE, startY + i * CELL_SIZE, CELL_SIZE, CELL_SIZE, cellColor);
            }
        }

        DrawText("Draw a digit and press SPACE to recognize", 10, 10, 20, DARKGRAY);
        DrawText("Press C to clear", 10, 40, 20, DARKGRAY);

        if (recognizedDigit != -1) {
            char result[30];
            sprintf(result, "Recognized digit: %d", recognizedDigit);
            DrawText(result, 10, SCREEN_HEIGHT - 120, 20, DARKGRAY);
        }

        if (isDigitRecognized) {
            DrawRectangleRec(correctButton, LIGHTGRAY);
            DrawRectangleRec(incorrectButton, LIGHTGRAY);
            DrawText("Correct", correctButton.x + 20, correctButton.y + 5, 20, BLACK);
            DrawText("Incorrect", incorrectButton.x + 20, incorrectButton.y + 5, 20, BLACK);

            if (isCorrect) {
                DrawText("User confirmed: Correct", 10, SCREEN_HEIGHT - 40, 20, GREEN);
            } else if (!isCorrect && isDigitRecognized) {
                DrawText("User confirmed: Incorrect", 10, SCREEN_HEIGHT - 40, 20, RED);
            }
        }

        EndDrawing();
    }

    NN_destroy(nn);
    CloseWindow();

    return 0;
}

void InitializeNN(NN_t **nn) {
    size_t layers[] = {GRID_SIZE * GRID_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 0};

    ActivationFunction activationFunctions[NUM_LAYERS] = {SIGMOID, SIGMOID, SIGMOID};
    ActivationDerivative activationDerivatives[NUM_LAYERS] = {SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE, SIGMOID_DERIVATIVE}; 

    *nn = NN_init(layers, activationFunctions, activationDerivatives, MSE, MSE_DERIVATIVE);
}

void Draw_Grid(int startX, int startY) {
    for (int i = 0; i <= GRID_SIZE; i++) {
        DrawLine(startX + i * CELL_SIZE, startY, startX + i * CELL_SIZE, startY + DRAWING_AREA_SIZE, LIGHTGRAY);
        DrawLine(startX, startY + i * CELL_SIZE, startX + DRAWING_AREA_SIZE, startY + i * CELL_SIZE, LIGHTGRAY);
    }
}

void UpdateDrawing(int startX, int startY, long double *input) {
    Vector2 mousePos = GetMousePosition();
    int gridX = (mousePos.x - startX) / CELL_SIZE;
    int gridY = (mousePos.y - startY) / CELL_SIZE;

    if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
        int index = gridY * GRID_SIZE + gridX;
        input[index] = 1.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = gridX + dx;
                int ny = gridY + dy;
                if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                    int neighbor_index = ny * GRID_SIZE + nx;
                    input[neighbor_index] = fmax(input[neighbor_index], 0.5);
                }
            }
        }
    }
}

int RecognizeDigit(NN_t *nn, long double *input) {
    long double *y_pred = NN_forward(nn, input);

    int recognized_digit = 0;
    long double max_prob = y_pred[0];

    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (y_pred[i] > max_prob) {
            max_prob = y_pred[i];
            recognized_digit = i;
        }
    }

    long double y_true[OUTPUT_SIZE] = {0};
    y_true[recognized_digit] = 1.0;

    NN_backprop(nn, input, y_true, *y_pred);

    printf("Recognized digit: %d\n", recognized_digit);
    free(y_pred);  
    return recognized_digit;
}

