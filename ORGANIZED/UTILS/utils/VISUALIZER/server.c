#include "nn_protocol.h"
#include "NN_visualizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include "raylib.h"

#define PORT 8080
#define BUFFER_SIZE 65536

static int server_socket = -1;
static int client_socket = -1;
static struct sockaddr_in server_addr, client_addr;
static char recv_buffer[BUFFER_SIZE];
static NN_t* current_nn = NULL;

int main(void) {
    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        return -1;
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("Setsockopt failed");
        return -1;
    }

    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind socket
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        return -1;
    }

    // Listen for connections
    if (listen(server_socket, 1) < 0) {
        perror("Listen failed");
        return -1;
    }

    printf("Server started on port %d\n", PORT);

    // Initialize visualizer
    InitWindow(VIS_WINDOW_WIDTH, VIS_WINDOW_HEIGHT, "Neural Network Visualization");
    SetTargetFPS(60);

    // Set server socket to non-blocking
    int flags = fcntl(server_socket, F_GETFL, 0);
    fcntl(server_socket, F_SETFL, flags | O_NONBLOCK);

    // Main loop
    while (!WindowShouldClose()) {
        // Accept new connections
        if (client_socket < 0) {
            socklen_t client_len = sizeof(client_addr);
            client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket >= 0) {
                // Set client socket to non-blocking
                flags = fcntl(client_socket, F_GETFL, 0);
                fcntl(client_socket, F_SETFL, flags | O_NONBLOCK);
                printf("Client connected\n");
            }
        }

        // Receive data
        if (client_socket >= 0) {
            ssize_t bytes = recv(client_socket, recv_buffer, BUFFER_SIZE, MSG_DONTWAIT);
            if (bytes > 0) {
                // Free previous network if it exists
                if (current_nn != NULL) {
                    // TODO: Implement NN_destroy
                    free(current_nn);
                }

                // Deserialize received network
                current_nn = DeserializeNN(recv_buffer, bytes);
                if (current_nn == NULL) {
                    printf("Failed to deserialize neural network\n");
                }
            } else if (bytes == 0 || (bytes < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                // Client disconnected or error
                close(client_socket);
                client_socket = -1;
                printf("Client disconnected\n");
            }
        }

        // Update visualization
        BeginDrawing();
        ClearBackground(RAYWHITE);  
        if (current_nn != NULL) {
            DrawNeuralNetwork(current_nn);
        } else {
            DrawText("Waiting for neural network data...", 10, 10, 20, GRAY);
        }
        EndDrawing();
    }

    // Cleanup
    if (client_socket >= 0) {
        close(client_socket);
    }
    close(server_socket);
    if (current_nn != NULL) {
        // TODO: Implement NN_destroy
        free(current_nn);
    }
    CloseWindow();

    return 0;
}
