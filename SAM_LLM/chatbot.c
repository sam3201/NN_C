#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../utils/Raylib/src/raylib.h"
#include "../SAM/SAM.h"

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define MAX_INPUT_CHARS 256
#define MAX_MESSAGES 100
#define MAX_MESSAGE_LENGTH 500

// Message structure
typedef struct {
    char text[MAX_MESSAGE_LENGTH];
    bool is_user;
    time_t timestamp;
} Message;

// Chat history
Message messages[MAX_MESSAGES];
int message_count = 0;
int scroll_offset = 0;

// Input box
char input_text[MAX_INPUT_CHARS + 1] = "\0";
int input_letter_count = 0;
Rectangle input_box = { 20, SCREEN_HEIGHT - 80, SCREEN_WIDTH - 120, 50 };
bool mouse_on_input = false;
int input_frames_counter = 0;

// SAM model
SAM_t* sam_model = NULL;
bool model_loaded = false;

// Simple character-level encoding/decoding
void encode_text(const char* text, long double* encoded, size_t max_len) {
    size_t len = strlen(text);
    if (len > max_len) len = max_len;
    
    for (size_t i = 0; i < max_len; i++) {
        if (i < len) {
            encoded[i] = (long double)((unsigned char)text[i]) / 255.0L;
        } else {
            encoded[i] = 0.0L;
        }
    }
}

void decode_text(const long double* encoded, char* decoded, size_t max_len) {
    for (size_t i = 0; i < max_len; i++) {
        int ascii = (int)(encoded[i] * 255.0L);
        if (ascii >= 32 && ascii <= 126) {
            decoded[i] = (char)ascii;
        } else if (ascii == 10) {
            decoded[i] = '\n';
        } else {
            decoded[i] = ' ';
        }
    }
    decoded[max_len] = '\0';
}

// Generate response using SAM model
void generate_response(const char* user_input, char* response) {
    if (!model_loaded || !sam_model) {
        strcpy(response, "Model not loaded. Please train a model first.");
        return;
    }
    
    // Encode user input
    long double* input = (long double*)malloc(256 * sizeof(long double));
    encode_text(user_input, input, 256);
    
    // Create input sequence
    long double** input_seq = (long double**)malloc(sizeof(long double*));
    input_seq[0] = input;
    
    // Get model response
    long double* output = SAM_forward(sam_model, input_seq, 1);
    
    if (output) {
        decode_text(output, response, 64);
        
        // Clean up response (remove extra spaces, ensure it's readable)
        size_t len = strlen(response);
        for (size_t i = 0; i < len; i++) {
            if (response[i] < 32 || response[i] > 126) {
                response[i] = ' ';
            }
        }
        
        // Trim response
        while (len > 0 && response[len - 1] == ' ') {
            response[len - 1] = '\0';
            len--;
        }
        
        free(output);
    } else {
        strcpy(response, "Error generating response.");
    }
    
    free(input);
    free(input_seq);
}

// Add message to chat history
void add_message(const char* text, bool is_user) {
    if (message_count >= MAX_MESSAGES) {
        // Shift messages up
        for (int i = 0; i < MAX_MESSAGES - 1; i++) {
            messages[i] = messages[i + 1];
        }
        message_count = MAX_MESSAGES - 1;
    }
    
    strncpy(messages[message_count].text, text, MAX_MESSAGE_LENGTH - 1);
    messages[message_count].text[MAX_MESSAGE_LENGTH - 1] = '\0';
    messages[message_count].is_user = is_user;
    messages[message_count].timestamp = time(NULL);
    message_count++;
    
    // Auto-scroll to bottom
    scroll_offset = 0;
}

// Draw chat messages
void draw_messages(void) {
    int start_y = 20;
    int line_height = 30;
    int max_visible = (SCREEN_HEIGHT - 120) / line_height;
    
    int start_idx = (message_count > max_visible) ? message_count - max_visible - scroll_offset : 0;
    if (start_idx < 0) start_idx = 0;
    
    for (int i = start_idx; i < message_count; i++) {
        int y = start_y + (i - start_idx) * line_height;
        
        // Draw message background
        Color bg_color = messages[i].is_user ? (Color){200, 220, 255, 255} : (Color){240, 240, 240, 255};
        DrawRectangle(10, y - 5, SCREEN_WIDTH - 20, line_height - 2, bg_color);
        
        // Draw message text
        Color text_color = messages[i].is_user ? DARKBLUE : DARKGRAY;
        const char* prefix = messages[i].is_user ? "You: " : "SAM: ";
        char display_text[600];
        snprintf(display_text, sizeof(display_text), "%s%s", prefix, messages[i].text);
        
        // Wrap text if too long
        int text_width = MeasureText(display_text, 18);
        if (text_width > SCREEN_WIDTH - 40) {
            // Simple truncation for now
            char truncated[600];
            strncpy(truncated, display_text, 80);
            truncated[80] = '\0';
            strcat(truncated, "...");
            DrawText(truncated, 15, y, 18, text_color);
        } else {
            DrawText(display_text, 15, y, 18, text_color);
        }
    }
}

int main(void) {
    // Initialize window
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "SAM LLM Chatbot");
    SetTargetFPS(60);
    
    // Try to load SAM model
    printf("Loading SAM model...\n");
    sam_model = SAM_load("../sam_trained_model.bin");
    if (!sam_model) {
        sam_model = SAM_load("../sam_hello_world.bin");
    }
    
    if (sam_model) {
        model_loaded = true;
        printf("Model loaded successfully!\n");
        add_message("SAM model loaded. How can I help you?", false);
    } else {
        printf("No model found. Using placeholder responses.\n");
        add_message("No trained model found. Please train a model first.", false);
        add_message("You can still chat, but responses will be placeholder.", false);
    }
    
    // Main loop
    while (!WindowShouldClose()) {
        // Update
        // Check if mouse is on input box
        if (CheckCollisionPointRec(GetMousePosition(), input_box)) {
            mouse_on_input = true;
        } else {
            mouse_on_input = false;
        }
        
        // Handle text input
        if (mouse_on_input) {
            SetMouseCursor(MOUSE_CURSOR_IBEAM);
            
            int key = GetCharPressed();
            while (key > 0) {
                if ((key >= 32) && (key <= 126) && (input_letter_count < MAX_INPUT_CHARS)) {
                    input_text[input_letter_count] = (char)key;
                    input_text[input_letter_count + 1] = '\0';
                    input_letter_count++;
                }
                key = GetCharPressed();
            }
            
            if (IsKeyPressed(KEY_BACKSPACE)) {
                input_letter_count--;
                if (input_letter_count < 0) input_letter_count = 0;
                input_text[input_letter_count] = '\0';
            }
            
            input_frames_counter++;
        } else {
            SetMouseCursor(MOUSE_CURSOR_DEFAULT);
            input_frames_counter = 0;
        }
        
        // Send message on Enter
        if (IsKeyPressed(KEY_ENTER) && input_letter_count > 0) {
            // Add user message
            add_message(input_text, true);
            
            // Generate response
            char response[MAX_MESSAGE_LENGTH];
            generate_response(input_text, response);
            
            // Add bot response
            add_message(response, false);
            
            // Clear input
            input_text[0] = '\0';
            input_letter_count = 0;
        }
        
        // Scroll with mouse wheel
        int wheel_move = GetMouseWheelMove();
        if (wheel_move != 0) {
            scroll_offset += wheel_move;
            if (scroll_offset < 0) scroll_offset = 0;
            int max_scroll = message_count - ((SCREEN_HEIGHT - 120) / 30);
            if (scroll_offset > max_scroll && max_scroll > 0) scroll_offset = max_scroll;
        }
        
        // Draw
        BeginDrawing();
        
        ClearBackground(RAYWHITE);
        
        // Draw title
        DrawText("SAM LLM Chatbot", 20, 5, 24, DARKGRAY);
        DrawLine(0, 35, SCREEN_WIDTH, 35, LIGHTGRAY);
        
        // Draw messages
        draw_messages();
        
        // Draw input box
        DrawRectangleRec(input_box, LIGHTGRAY);
        if (mouse_on_input) {
            DrawRectangleLines((int)input_box.x, (int)input_box.y, 
                             (int)input_box.width, (int)input_box.height, BLUE);
        } else {
            DrawRectangleLines((int)input_box.x, (int)input_box.y, 
                             (int)input_box.width, (int)input_box.height, DARKGRAY);
        }
        
        DrawText(input_text, (int)input_box.x + 5, (int)input_box.y + 15, 20, MAROON);
        
        // Draw blinking cursor
        if (mouse_on_input && (input_frames_counter / 20) % 2 == 0) {
            int text_width = MeasureText(input_text, 20);
            DrawText("_", (int)input_box.x + 10 + text_width, (int)input_box.y + 15, 20, MAROON);
        }
        
        // Draw send button
        Rectangle send_button = { SCREEN_WIDTH - 90, SCREEN_HEIGHT - 80, 70, 50 };
        bool mouse_on_send = CheckCollisionPointRec(GetMousePosition(), send_button);
        Color send_color = mouse_on_send ? BLUE : DARKBLUE;
        DrawRectangleRec(send_button, send_color);
        DrawText("Send", SCREEN_WIDTH - 75, SCREEN_HEIGHT - 65, 20, WHITE);
        
        // Handle send button click
        if (mouse_on_send && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && input_letter_count > 0) {
            add_message(input_text, true);
            char response[MAX_MESSAGE_LENGTH];
            generate_response(input_text, response);
            add_message(response, false);
            input_text[0] = '\0';
            input_letter_count = 0;
        }
        
        // Draw status
        if (model_loaded) {
            DrawText("Model: Loaded", SCREEN_WIDTH - 150, 5, 16, GREEN);
        } else {
            DrawText("Model: Not Loaded", SCREEN_WIDTH - 180, 5, 16, RED);
        }
        
        EndDrawing();
    }
    
    // Cleanup
    if (sam_model) {
        SAM_destroy(sam_model);
    }
    CloseWindow();
    
    return 0;
}

