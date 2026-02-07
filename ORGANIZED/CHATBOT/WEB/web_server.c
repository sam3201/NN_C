#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <fcntl.h>

#define PORT 8080
#define BUFFER_SIZE 4096
#define WEB_ROOT "."

// HTTP response headers
const char* HTTP_OK = "HTTP/1.1 200 OK\r\n";
const char* HTTP_NOT_FOUND = "HTTP/1.1 404 Not Found\r\n";
const char* CONTENT_TYPE_HTML = "Content-Type: text/html\r\n";
const char* CONTENT_TYPE_CSS = "Content-Type: text/css\r\n";
const char* CONTENT_TYPE_JS = "Content-Type: application/javascript\r\n";
const char* CONTENT_TYPE_TEXT = "Content-Type: text/plain\r\n";
const char* CONNECTION_CLOSE = "Connection: close\r\n\r\n";

// Get content type based on file extension
const char* get_content_type(const char* filename) {
    if (strstr(filename, ".html") || strstr(filename, ".htm")) {
        return CONTENT_TYPE_HTML;
    } else if (strstr(filename, ".css")) {
        return CONTENT_TYPE_CSS;
    } else if (strstr(filename, ".js")) {
        return CONTENT_TYPE_JS;
    } else if (strstr(filename, ".txt") || strstr(filename, ".md")) {
        return CONTENT_TYPE_TEXT;
    }
    return CONTENT_TYPE_HTML; // Default to HTML
}

// Read file content
char* read_file(const char* filename, size_t* file_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char* buffer = malloc(*file_size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    
    // Read file
    size_t bytes_read = fread(buffer, 1, *file_size, file);
    buffer[bytes_read] = '\0';
    
    fclose(file);
    return buffer;
}

// Send HTTP response
void send_response(int client_socket, const char* status, const char* content_type, const char* content, size_t content_length) {
    char response_header[BUFFER_SIZE];
    snprintf(response_header, sizeof(response_header), 
             "%s%sContent-Length: %zu\r\n%s", 
             status, content_type, content_length, CONNECTION_CLOSE);
    
    send(client_socket, response_header, strlen(response_header), 0);
    send(client_socket, content, content_length, 0);
}

// Send 404 response
void send_404(int client_socket) {
    const char* not_found_content = 
        "<html><body><h1>404 Not Found</h1><p>The requested resource was not found.</p></body></html>";
    
    send_response(client_socket, HTTP_NOT_FOUND, CONTENT_TYPE_HTML, 
                 not_found_content, strlen(not_found_content));
}

// Handle HTTP request
void handle_request(int client_socket, const char* request) {
    char method[16], path[256], version[16];
    char filename[512];
    
    // Parse HTTP request
    if (sscanf(request, "%15s %255s %15s", method, path, version) != 3) {
        send_404(client_socket);
        return;
    }
    
    // Only handle GET requests
    if (strcmp(method, "GET") != 0) {
        send_404(client_socket);
        return;
    }
    
    // Construct file path
    if (strcmp(path, "/") == 0) {
        strcpy(filename, "web_chatbot.html");
    } else {
        // Remove leading slash
        const char* path_start = path;
        if (path[0] == '/') {
            path_start = path + 1;
        }
        snprintf(filename, sizeof(filename), "%s/%s", WEB_ROOT, path_start);
    }
    
    // Read file
    size_t file_size;
    char* file_content = read_file(filename, &file_size);
    
    if (file_content) {
        // Send file content
        send_response(client_socket, HTTP_OK, get_content_type(filename), 
                     file_content, file_size);
        free(file_content);
    } else {
        // Send 404
        send_404(client_socket);
    }
}

// Main server loop
int main() {
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];
    
    printf("=== AGI Chatbot Web Server ===\n");
    printf("Starting server on port %d...\n", PORT);
    
    // Create socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        return 1;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Configure server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    // Bind socket
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_socket);
        return 1;
    }
    
    // Listen for connections
    if (listen(server_socket, 10) < 0) {
        perror("Listen failed");
        close(server_socket);
        return 1;
    }
    
    printf("Server started successfully!\n");
    printf("Open your browser and go to: http://localhost:%d\n", PORT);
    printf("Press Ctrl+C to stop the server\n\n");
    
    // Main server loop
    while (1) {
        // Accept connection
        client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket < 0) {
            perror("Accept failed");
            continue;
        }
        
        // Get client IP
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        
        // Receive request
        ssize_t bytes_received = recv(client_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            
            printf("Request from %s:%d - %s\n", 
                   client_ip, ntohs(client_addr.sin_port), buffer);
            
            // Handle request
            handle_request(client_socket, buffer);
        }
        
        // Close client socket
        close(client_socket);
    }
    
    // Close server socket
    close(server_socket);
    return 0;
}
