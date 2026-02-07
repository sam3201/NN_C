#include "client.h"
#include "nn_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

#define SERVER_PORT 8080
#define BUFFER_SIZE 65536

static int client_socket = -1;
static struct sockaddr_in server_addr;
static char send_buffer[BUFFER_SIZE];

int InitializeVisualizerClient(void) {
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        perror("Socket creation failed");
        return -1;
    }

    // Set socket to non-blocking
    int flags = fcntl(client_socket, F_GETFL, 0);
    fcntl(client_socket, F_SETFL, flags | O_NONBLOCK);

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr) <= 0) {
        perror("Invalid address");
        close(client_socket);
        return -1;
    }

    // Non-blocking connect
    connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr));
    return 0;
}

void SendNetworkToVisualizer(NN_t* nn) {
    if (client_socket < 0) return;
    
    // Clear send buffer
    memset(send_buffer, 0, BUFFER_SIZE);
    
    // Serialize neural network
    int bytes = SerializeNN(nn, send_buffer, BUFFER_SIZE);
    if (bytes <= 0) {
        printf("Failed to serialize neural network\n");
        return;
    }
    
    // Send data
    ssize_t sent = send(client_socket, send_buffer, bytes, MSG_NOSIGNAL);
    if (sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        printf("Failed to send data to visualizer\n");
    }
}

void CloseVisualizerClient(void) {
    if (client_socket >= 0) {
        close(client_socket);
        client_socket = -1;
    }
}
