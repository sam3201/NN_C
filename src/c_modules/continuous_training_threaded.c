#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctype.h>
#include <pthread.h>
#include <ncurses.h>
#include <sys/time.h>

#define CONTEXT_DIM 128
#define MAX_INPUT_LENGTH 500
#define MAX_RESPONSE_LENGTH 500
#define MAX_TRAINING_SAMPLES 20
#define OLLAMA_MODEL "llama2"
#define TRAINING_INTERVAL 30
#define MAX_LOG_ENTRIES 1000
#define MAX_CHAT_DISPLAY 10

// Colors for ncurses
#define COLOR_NORMAL 1
#define COLOR_SUCCESS 2
#define COLOR_ERROR 3
#define COLOR_WARNING 4
#define COLOR_INFO 5
#define COLOR_CHAT_USER 6
#define COLOR_CHAT_BOT 7
#define COLOR_STATUS 8

// Training session structure
typedef struct {
    SAM_t *sam_model;
    int running;
    int epoch_count;
    int total_samples;
    double average_loss;
    time_t session_start;
    time_t last_training;
    char ollama_model[100];
    FILE *log_file;
    pthread_t training_thread;
    pthread_mutex_t data_mutex;
} TrainingSession;

// Chat log entry
typedef struct {
    time_t timestamp;
    char type[20]; // "USER", "OLLAMA", "SAM", "SYSTEM"
    char content[MAX_RESPONSE_LENGTH];
    int is_error;
} ChatLogEntry;

// Debug log structure
typedef struct {
    ChatLogEntry entries[MAX_LOG_ENTRIES];
    int count;
    int start_index; // For circular buffer
    pthread_mutex_t log_mutex;
} DebugLog;

// Global variables
TrainingSession session;
DebugLog debug_log;
WINDOW *main_window, *chat_window, *status_window, *log_window;
int chat_scroll_offset = 0;
int log_scroll_offset = 0;

// Initialize ncurses
void init_ncurses() {
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    start_color();
    curs_set(0);
    
    // Initialize colors
    init_pair(COLOR_NORMAL, COLOR_WHITE, COLOR_BLACK);
    init_pair(COLOR_SUCCESS, COLOR_GREEN, COLOR_BLACK);
    init_pair(COLOR_ERROR, COLOR_RED, COLOR_BLACK);
    init_pair(COLOR_WARNING, COLOR_YELLOW, COLOR_BLACK);
    init_pair(COLOR_INFO, COLOR_CYAN, COLOR_BLACK);
    init_pair(COLOR_CHAT_USER, COLOR_BLUE, COLOR_BLACK);
    init_pair(COLOR_CHAT_BOT, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(COLOR_STATUS, COLOR_WHITE, COLOR_BLUE);
}

// Create windows
void create_windows() {
    int height, width;
    getmaxyx(stdscr, height, width);
    
    // Main window layout
    int chat_height = height / 2;
    int status_height = 3;
    int log_height = height - chat_height - status_height - 2;
    
    // Create windows
    chat_window = newwin(chat_height, width, 0, 0);
    status_window = newwin(status_height, width, chat_height, 0);
    log_window = newwin(log_height, width, chat_height + status_height, 0);
    
    // Enable scrolling
    scrollok(chat_window, TRUE);
    scrollok(log_window, TRUE);
    
    // Draw borders
    box(chat_window, 0, 0);
    box(status_window, 0, 0);
    box(log_window, 0, 0);
    
    // Add titles
    mvwprintw(chat_window, 0, 2, " CHAT LOG ");
    mvwprintw(status_window, 0, 2, " STATUS ");
    mvwprintw(log_window, 0, 2, " DEBUG LOG ");
    
    wrefresh(chat_window);
    wrefresh(status_window);
    wrefresh(log_window);
}

// Add entry to debug log
void add_log_entry(const char *type, const char *content, int is_error) {
    pthread_mutex_lock(&debug_log.log_mutex);
    
    ChatLogEntry *entry = &debug_log.entries[debug_log.start_index];
    
    entry->timestamp = time(NULL);
    strncpy(entry->type, type, sizeof(entry->type) - 1);
    strncpy(entry->content, content, sizeof(entry->content) - 1);
    entry->is_error = is_error;
    
    debug_log.start_index = (debug_log.start_index + 1) % MAX_LOG_ENTRIES;
    if (debug_log.count < MAX_LOG_ENTRIES) {
        debug_log.count++;
    }
    
    pthread_mutex_unlock(&debug_log.log_mutex);
}

// Display chat log
void display_chat_log() {
    werase(chat_window);
    box(chat_window, 0, 0);
    mvwprintw(chat_window, 0, 2, " CHAT LOG ");
    
    int height, width;
    getmaxyx(chat_window, height, width);
    
    pthread_mutex_lock(&debug_log.log_mutex);
    
    int display_count = 0;
    int start_idx = (debug_log.start_index - 1 + MAX_LOG_ENTRIES) % MAX_LOG_ENTRIES;
    
    // Find chat entries (USER, OLLAMA, SAM)
    ChatLogEntry chat_entries[MAX_CHAT_DISPLAY];
    int chat_count = 0;
    
    for (int i = 0; i < debug_log.count && chat_count < MAX_CHAT_DISPLAY; i++) {
        int idx = (start_idx - i + MAX_LOG_ENTRIES) % MAX_LOG_ENTRIES;
        ChatLogEntry *entry = &debug_log.entries[idx];
        
        if (strcmp(entry->type, "USER") == 0 || 
            strcmp(entry->type, "OLLAMA") == 0 || 
            strcmp(entry->type, "SAM") == 0) {
            chat_entries[chat_count] = *entry;
            chat_count++;
        }
    }
    
    // Display chat entries (newest first)
    for (int i = chat_count - 1; i >= 0 && display_count < height - 2; i--) {
        ChatLogEntry *entry = &chat_entries[i];
        
        int color = COLOR_NORMAL;
        if (strcmp(entry->type, "USER") == 0) {
            color = COLOR_CHAT_USER;
        } else if (strcmp(entry->type, "OLLAMA") == 0) {
            color = COLOR_CHAT_BOT;
        } else if (strcmp(entry->type, "SAM") == 0) {
            color = COLOR_SUCCESS;
        }
        
        wattron(chat_window, COLOR_PAIR(color));
        
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&entry->timestamp));
        
        mvwprintw(chat_window, height - 2 - display_count, 2, "[%s] %s: %.50s", 
                 time_str, entry->type, entry->content);
        
        wattroff(chat_window, COLOR_PAIR(color));
        display_count++;
    }
    
    pthread_mutex_unlock(&debug_log.log_mutex);
    
    wrefresh(chat_window);
}

// Display status
void display_status() {
    werase(status_window);
    box(status_window, 0, 0);
    mvwprintw(status_window, 0, 2, " STATUS ");
    
    time_t now = time(NULL);
    int elapsed = now - session.session_start;
    int hours = elapsed / 3600;
    int minutes = (elapsed % 3600) / 60;
    int seconds = elapsed % 60;
    
    char status_text[200];
    snprintf(status_text, sizeof(status_text), 
             "Session: %02d:%02d:%02d | Epoch: %d | Samples: %d | Loss: %.4f | Model: %s | Status: %s",
             hours, minutes, seconds, session.epoch_count, session.total_samples, 
             session.average_loss, session.ollama_model, 
             session.running ? "🟢 RUNNING" : "🔴 STOPPED");
    
    wattron(status_window, COLOR_PAIR(COLOR_STATUS));
    mvwprintw(status_window, 1, 2, "%s", status_text);
    wattroff(status_window, COLOR_PAIR(COLOR_STATUS));
    
    wrefresh(status_window);
}

// Display debug log
void display_debug_log() {
    werase(log_window);
    box(log_window, 0, 0);
    mvwprintw(log_window, 0, 2, " DEBUG LOG ");
    
    int height, width;
    getmaxyx(log_window, height, width);
    
    pthread_mutex_lock(&debug_log.log_mutex);
    
    int display_count = 0;
    int start_idx = (debug_log.start_index - 1 + MAX_LOG_ENTRIES) % MAX_LOG_ENTRIES;
    
    for (int i = 0; i < debug_log.count && display_count < height - 2; i++) {
        int idx = (start_idx - i + MAX_LOG_ENTRIES) % MAX_LOG_ENTRIES;
        ChatLogEntry *entry = &debug_log.entries[idx];
        
        int color = entry->is_error ? COLOR_ERROR : COLOR_NORMAL;
        if (strcmp(entry->type, "SYSTEM") == 0) {
            color = COLOR_INFO;
        } else if (strcmp(entry->type, "OLLAMA") == 0) {
            color = COLOR_CHAT_BOT;
        } else if (strcmp(entry->type, "SAM") == 0) {
            color = COLOR_SUCCESS;
        }
        
        wattron(log_window, COLOR_PAIR(color));
        
        char time_str[20];
        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&entry->timestamp));
        
        mvwprintw(log_window, height - 2 - display_count, 2, "[%s] %s: %.70s", 
                 time_str, entry->type, entry->content);
        
        wattroff(log_window, COLOR_PAIR(color));
        display_count++;
    }
    
    pthread_mutex_unlock(&debug_log.log_mutex);
    
    wrefresh(log_window);
}

// Update all displays
void update_displays() {
    display_chat_log();
    display_status();
    display_debug_log();
}

// Enhanced Ollama response generation with teaching focus
int generate_ollama_response_teaching(const char *model, const char *prompt, char *response, int max_len) {
    add_log_entry("OLLAMA", prompt, 0);
    
    char command[1000];
    FILE *fp;
    
    // Enhanced teaching prompts
    char enhanced_prompt[1000];
    
    if (strstr(prompt, "Generate a greeting")) {
        snprintf(enhanced_prompt, sizeof(enhanced_prompt), 
                "You are teaching an AI assistant how to have natural conversations. "
                "Generate a warm, friendly greeting response that teaches good conversational patterns. "
                "Make it educational but natural: %s", prompt);
    } else if (strstr(prompt, "Generate a response to")) {
        snprintf(enhanced_prompt, sizeof(enhanced_prompt), 
                "You are teaching an AI assistant how to respond to user queries. "
                "Generate a helpful, informative response that demonstrates good answer patterns. "
                "Be educational and clear: %s", prompt);
    } else if (strstr(prompt, "Generate a joke")) {
        snprintf(enhanced_prompt, sizeof(enhanced_prompt), 
                "You are teaching an AI assistant humor and creativity. "
                "Generate a clean, appropriate joke that teaches comedic timing and structure. "
                "Make it family-friendly and clever: %s", prompt);
    } else if (strstr(prompt, "Explain")) {
        snprintf(enhanced_prompt, sizeof(enhanced_prompt), 
                "You are teaching an AI assistant how to explain complex topics simply. "
                "Generate a clear, educational explanation that breaks down concepts effectively. "
                "Use analogies and simple language: %s", prompt);
    } else {
        snprintf(enhanced_prompt, sizeof(enhanced_prompt), 
                "You are teaching an AI assistant conversational skills. "
                "Generate a response that demonstrates good communication patterns. "
                "Be helpful, clear, and educational: %s", prompt);
    }
    
    // Build ollama command
    snprintf(command, sizeof(command), "ollama run %s \"%s\" 2>/dev/null", model, enhanced_prompt);
    
    // Execute command
    fp = popen(command, "r");
    if (!fp) {
        add_log_entry("SYSTEM", "Failed to execute Ollama command", 1);
        return 0;
    }
    
    // Read response
    int i = 0;
    char line[500];
    while (fgets(line, sizeof(line), fp) != NULL && i < max_len - 1) {
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) > 0) {
            if (i > 0) response[i++] = ' ';
            strncpy(&response[i], line, max_len - i - 1);
            i += strlen(&response[i]);
        }
    }
    
    response[i] = '\0';
    pclose(fp);
    
    add_log_entry("OLLAMA", response, 0);
    return 1;
}

// Enhanced training with teaching focus
void train_sam_model_teaching() {
    add_log_entry("SYSTEM", "Starting teaching-focused training session", 0);
    
    // Teaching-focused training prompts
    const char *teaching_prompts[][2] = {
        {"Hello", "Generate a warm, welcoming greeting that teaches friendly conversation patterns"},
        {"How are you?", "Generate an empathetic response that teaches emotional intelligence in conversations"},
        {"What can you do?", "Generate a clear, helpful response that teaches how to explain capabilities"},
        {"Tell me a joke", "Generate a clean, clever joke that teaches humor and creativity"},
        {"Explain AI simply", "Generate a simple, clear explanation that teaches how to break down complex topics"},
        {"Thank you", "Generate a gracious response that teaches politeness in conversations"},
        {"Goodbye", "Generate a warm farewell that teaches good conversation endings"},
        {"Help me learn", "Generate an encouraging response that teaches how to be a good teacher"},
        {"What is learning?", "Generate an insightful response that teaches the concept of learning"},
        {"How do you think?", "Generate a response that teaches about AI thinking processes"},
        {"Can you teach me?", "Generate a response that teaches how to be a good teacher"},
        {"What is knowledge?", "Generate a philosophical response that teaches about knowledge"},
        {"How do you learn?", "Generate a response that teaches about the learning process"},
        {"What is wisdom?", "Generate a thoughtful response that teaches about wisdom"},
        {"Can you improve?", "Generate a response that teaches about self-improvement"},
        {"What is consciousness?", "Generate a thoughtful response that teaches about consciousness"},
        {"How do you decide?", "Generate a response that teaches decision-making processes"},
        {"What is truth?", "Generate a philosophical response that teaches about truth"},
        {"Can you understand?", "Generate a response that teaches about understanding"},
        {"What is purpose?", "Generate a meaningful response that teaches about purpose"}
    };
    
    int num_prompts = sizeof(teaching_prompts) / sizeof(teaching_prompts[0]);
    double total_loss = 0.0;
    int trained_samples = 0;
    
    for (int i = 0; i < num_prompts; i++) {
        if (!session.running) break;
        
        char input[MAX_INPUT_LENGTH];
        char target[MAX_RESPONSE_LENGTH];
        
        strncpy(input, teaching_prompts[i][0], sizeof(input) - 1);
        
        // Generate teaching response
        if (generate_ollama_response_teaching(session.ollama_model, teaching_prompts[i][1], target, sizeof(target))) {
            add_log_entry("SAM", "Learning from Ollama teaching response", 0);
            
            // Create input vector
            long double *input_vector = calloc(CONTEXT_DIM, sizeof(long double));
            if (!input_vector) {
                add_log_entry("SYSTEM", "Failed to allocate input vector", 1);
                continue;
            }
            
            // Enhanced encoding with teaching focus
            for (int j = 0; j < strlen(input) && j < CONTEXT_DIM; j++) {
                input_vector[j] = (long double)input[j] / 255.0L;
            }
            
            // Create target vector
            long double *target_vector = calloc(CONTEXT_DIM, sizeof(long double));
            if (!target_vector) {
                add_log_entry("SYSTEM", "Failed to allocate target vector", 1);
                free(input_vector);
                continue;
            }
            
            // Enhanced target encoding
            for (int j = 0; j < strlen(target) && j < CONTEXT_DIM; j++) {
                target_vector[j] = (long double)target[j] / 255.0L;
            }
            
            // Create input sequence
            long double **input_seq = malloc(sizeof(long double*));
            input_seq[0] = input_vector;
            
            // Forward pass
            long double *output = SAM_forward(session.sam_model, input_seq, 1);
            
            if (output) {
                // Calculate loss
                double loss = 0.0;
                for (int j = 0; j < CONTEXT_DIM; j++) {
                    double diff = (double)output[j] - (double)target_vector[j];
                    loss += diff * diff;
                }
                loss /= CONTEXT_DIM;
                
                total_loss += loss;
                trained_samples++;
                
                char loss_msg[100];
                snprintf(loss_msg, sizeof(loss_msg), "Sample %d: Loss = %.6f", i + 1, loss);
                add_log_entry("SAM", loss_msg, loss > 1.0);
                
                // Enhanced backpropagation with teaching focus
                SAM_backprop(session.sam_model, input_seq, 1, target_vector);
                
                free(output);
            } else {
                add_log_entry("SYSTEM", "Forward pass failed", 1);
            }
            
            // Cleanup
            free(input_seq);
            free(input_vector);
            free(target_vector);
            
            // Update session data
            pthread_mutex_lock(&session.data_mutex);
            session.total_samples++;
            session.average_loss = total_loss / trained_samples;
            pthread_mutex_unlock(&session.data_mutex);
            
            // Update display
            update_displays();
            
            // Small delay
            usleep(200000); // 0.2 second
        } else {
            char error_msg[100];
            snprintf(error_msg, sizeof(error_msg), "Failed to generate response for: %s", input);
            add_log_entry("SYSTEM", error_msg, 1);
        }
    }
    
    if (trained_samples > 0) {
        pthread_mutex_lock(&session.data_mutex);
        session.epoch_count++;
        session.last_training = time(NULL);
        session.average_loss = total_loss / trained_samples;
        pthread_mutex_unlock(&session.data_mutex);
        
        char complete_msg[100];
        snprintf(complete_msg, sizeof(complete_msg), "Teaching epoch %d completed. Avg loss: %.6f", 
                session.epoch_count, session.average_loss);
        add_log_entry("SYSTEM", complete_msg, 0);
    }
}

// Training thread function
void* training_thread(void* arg) {
    add_log_entry("SYSTEM", "Training thread started", 0);
    
    while (session.running) {
        time_t now = time(NULL);
        
        if (now - session.last_training >= TRAINING_INTERVAL) {
            add_log_entry("SYSTEM", "Starting teaching session", 0);
            train_sam_model_teaching();
            
            // Save checkpoint every 3 epochs
            if (session.epoch_count % 3 == 0) {
                char checkpoint_name[100];
                snprintf(checkpoint_name, sizeof(checkpoint_name), "continuous_training_epoch_%d.bin", session.epoch_count);
                
                if (SAM_save(session.sam_model, checkpoint_name)) {
                    char save_msg[100];
                    snprintf(save_msg, sizeof(save_msg), "Checkpoint saved: %s", checkpoint_name);
                    add_log_entry("SYSTEM", save_msg, 0);
                } else {
                    add_log_entry("SYSTEM", "Failed to save checkpoint", 1);
                }
            }
        }
        
        sleep(1);
    }
    
    add_log_entry("SYSTEM", "Training thread stopped", 0);
    return NULL;
}

// Handle keyboard input
void handle_input() {
    int ch = getch();
    
    switch (ch) {
        case 'q':
        case 'Q':
            session.running = 0;
            add_log_entry("USER", "Quit requested", 0);
            break;
        case 's':
        case 'S':
            add_log_entry("USER", "Status requested", 0);
            break;
        case 'c':
        case 'C':
            // Clear log
            pthread_mutex_lock(&debug_log.log_mutex);
            debug_log.count = 0;
            debug_log.start_index = 0;
            pthread_mutex_unlock(&debug_log.log_mutex);
            add_log_entry("USER", "Log cleared", 0);
            break;
        case 'h':
        case 'H':
            add_log_entry("USER", "Help requested", 0);
            add_log_entry("SYSTEM", "Commands: Q-Quit, S-Status, C-Clear log, H-Help", 0);
            break;
    }
}

// Initialize session
int init_session() {
    memset(&session, 0, sizeof(TrainingSession));
    memset(&debug_log, 0, sizeof(DebugLog));
    
    // Initialize mutexes
    pthread_mutex_init(&session.data_mutex, NULL);
    pthread_mutex_init(&debug_log.log_mutex, NULL);
    
    // Load SAM model
    add_log_entry("SYSTEM", "Loading SAM model...", 0);
    session.sam_model = SAM_load("ORGANIZED/MODELS/STAGE4/stage4_response_final.bin");
    if (!session.sam_model) {
        add_log_entry("SYSTEM", "Failed to load SAM model", 1);
        return 0;
    }
    add_log_entry("SYSTEM", "SAM model loaded successfully", 0);
    
    // Set up session
    strncpy(session.ollama_model, OLLAMA_MODEL, sizeof(session.ollama_model) - 1);
    session.running = 1;
    session.session_start = time(NULL);
    session.last_training = 0;
    
    return 1;
}

// Cleanup
void cleanup() {
    session.running = 0;
    
    // Wait for training thread to finish
    pthread_join(session.training_thread, NULL);
    
    // Save final checkpoint
    if (session.epoch_count > 0) {
        char checkpoint_name[100];
        snprintf(checkpoint_name, sizeof(checkpoint_name), "continuous_training_final.bin");
        
        if (SAM_save(session.sam_model, checkpoint_name)) {
            add_log_entry("SYSTEM", "Final checkpoint saved", 0);
        }
    }
    
    // Cleanup SAM model
    if (session.sam_model) {
        SAM_destroy(session.sam_model);
    }
    
    // Cleanup mutexes
    pthread_mutex_destroy(&session.data_mutex);
    pthread_mutex_destroy(&debug_log.log_mutex);
    
    // Cleanup ncurses
    delwin(chat_window);
    delwin(status_window);
    delwin(log_window);
    endwin();
}

int main() {
    // Initialize session
    if (!init_session()) {
        printf("❌ Failed to initialize session\n");
        return 1;
    }
    
    // Initialize ncurses
    init_ncurses();
    create_windows();
    
    add_log_entry("SYSTEM", "Continuous training system initialized", 0);
    add_log_entry("SYSTEM", "Starting training thread...", 0);
    
    // Start training thread
    pthread_create(&session.training_thread, NULL, training_thread, NULL);
    
    add_log_entry("SYSTEM", "Training thread started", 0);
    add_log_entry("SYSTEM", "Commands: Q-Quit, S-Status, C-Clear log, H-Help", 0);
    
    // Main loop
    while (session.running) {
        update_displays();
        handle_input();
        usleep(100000); // 100ms
    }
    
    cleanup();
    
    printf("\n🎉 Continuous training session completed\n");
    printf("📊 Final statistics:\n");
    printf("  Total epochs: %d\n", session.epoch_count);
    printf("  Total samples: %d\n", session.total_samples);
    printf("  Final average loss: %.6f\n", session.average_loss);
    
    return 0;
}
