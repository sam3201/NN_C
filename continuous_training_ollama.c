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

#define CONTEXT_DIM 128
#define MAX_INPUT_LENGTH 500
#define MAX_RESPONSE_LENGTH 500
#define MAX_TRAINING_SAMPLES 100
#define OLLAMA_MODEL "llama2"  // Default Ollama model
#define TRAINING_INTERVAL 5 // Training every 30 seconds

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
} TrainingSession;

// Training sample
typedef struct {
    char input[MAX_INPUT_LENGTH];
    char target[MAX_RESPONSE_LENGTH];
    double loss;
    int processed;
} TrainingSample;

// Signal handler for graceful shutdown
volatile sig_atomic_t keep_running = 1;

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        keep_running = 0;
        printf("\n🛑 Received interrupt signal. Shutting down gracefully...\n");
    }
}

// Initialize signal handlers
void init_signal_handlers() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
}

// Check if Ollama is available
int check_ollama_available() {
    printf("🔍 Checking Ollama availability...\n");
    
    // Try to run ollama list command
    int result = system("ollama list > /dev/null 2>&1");
    
    if (result == 0) {
        printf("✅ Ollama is available\n");
        return 1;
    } else {
        printf("❌ Ollama is not available or not in PATH\n");
        printf("💡 Please install Ollama: https://ollama.ai/\n");
        return 0;
    }
}

// Generate response using Ollama
int generate_ollama_response(const char *model, const char *prompt, char *response, int max_len) {
    char command[1000];
    char temp_file[100];
    FILE *fp;
    
    // Create temporary file for the prompt
    snprintf(temp_file, sizeof(temp_file), "/tmp/ollama_prompt_%ld.txt", time(NULL));
    FILE *prompt_file = fopen(temp_file, "w");
    if (!prompt_file) {
        printf("❌ Failed to create temporary file\n");
        return 0;
    }
    
    fprintf(prompt_file, "%s", prompt);
    fclose(prompt_file);
    
    // Build ollama command
    snprintf(command, sizeof(command), "ollama run %s \"%s\" 2>/dev/null", model, prompt);
    
    // Execute command and capture output
    fp = popen(command, "r");
    if (!fp) {
        printf("❌ Failed to execute Ollama command\n");
        unlink(temp_file);
        return 0;
    }
    
    // Read response
    int i = 0;
    char line[500];
    while (fgets(line, sizeof(line), fp) != NULL && i < max_len - 1) {
        // Remove newline and add to response
        line[strcspn(line, "\n")] = '\0';
        if (strlen(line) > 0) {
            if (i > 0) response[i++] = ' ';
            strncpy(&response[i], line, max_len - i - 1);
            i += strlen(&response[i]);
        }
    }
    
    response[i] = '\0';
    pclose(fp);
    unlink(temp_file);
    
    return 1;
}

// Generate training data using Ollama
void generate_training_samples(TrainingSession *session, TrainingSample *samples, int *sample_count) {
    printf("🎯 Generating training samples using Ollama (%s)...\n", session->ollama_model);
    
    *sample_count = 0;
    
    // Training prompts for different types of responses
    const char *prompts[] = {
        "Generate a simple greeting response",
        "Generate a helpful response to 'How are you?'",
        "Generate a response to 'What can you do?'",
        "Generate a response to 'Tell me a joke'",
        "Generate a response to 'Explain AI simply'",
        "Generate a response to 'Thank you for helping'",
        "Generate a response to 'Goodbye'",
        "Generate a response to 'I need help with programming'",
        "Generate a response to 'What is machine learning?'",
        "Generate a response to 'How do you work?'"
    };
    
    const char *inputs[] = {
        "Hello",
        "How are you?",
        "What can you do?",
        "Tell me a joke",
        "Explain AI simply",
        "Thank you for helping",
        "Goodbye",
        "I need help with programming",
        "What is machine learning?",
        "How do you work?"
    };
    
    int num_prompts = sizeof(prompts) / sizeof(prompts[0]);
    
    for (int i = 0; i < num_prompts && *sample_count < MAX_TRAINING_SAMPLES; i++) {
        printf("  📝 Generating sample %d/%d...\n", i + 1, num_prompts);
        
        // Generate response using Ollama
        char response[MAX_RESPONSE_LENGTH];
        if (generate_ollama_response(session->ollama_model, prompts[i], response, sizeof(response))) {
            // Store training sample
            strcpy(samples[*sample_count].input, inputs[i]);
            strcpy(samples[*sample_count].target, response);
            samples[*sample_count].loss = 0.0;
            samples[*sample_count].processed = 0;
            (*sample_count)++;
            
            printf("    ✅ Generated: '%s' -> '%.50s...'\n", inputs[i], response);
        } else {
            printf("    ❌ Failed to generate response for prompt: %s\n", prompts[i]);
        }
        
        // Small delay to avoid overwhelming Ollama
        usleep(500000); // 0.5 second
    }
    
    printf("📊 Generated %d training samples\n", *sample_count);
}

// Train SAM model with generated samples
void train_sam_model(TrainingSession *session, TrainingSample *samples, int sample_count) {
    printf("🎓 Training SAM model with %d samples...\n", sample_count);
    
    if (!session->sam_model) {
        printf("❌ SAM model not initialized\n");
        return;
    }
    
    double total_loss = 0.0;
    int trained_samples = 0;
    
    for (int i = 0; i < sample_count; i++) {
        if (!samples[i].processed) {
            printf("  🔄 Training sample %d/%d: '%s'\n", i + 1, sample_count, samples[i].input);
            
            // Create input vector (simplified encoding)
            long double *input_vector = calloc(CONTEXT_DIM, sizeof(long double));
            if (!input_vector) {
                printf("    ❌ Failed to allocate input vector\n");
                continue;
            }
            
            // Simple character encoding
            for (int j = 0; j < strlen(samples[i].input) && j < CONTEXT_DIM; j++) {
                input_vector[j] = (long double)samples[i].input[j] / 255.0L;
            }
            
            // Create target vector (simplified encoding)
            long double *target_vector = calloc(CONTEXT_DIM, sizeof(long double));
            if (!target_vector) {
                printf("    ❌ Failed to allocate target vector\n");
                free(input_vector);
                continue;
            }
            
            for (int j = 0; j < strlen(samples[i].target) && j < CONTEXT_DIM; j++) {
                target_vector[j] = (long double)samples[i].target[j] / 255.0L;
            }
            
            // Create input sequence
            long double **input_seq = malloc(sizeof(long double*));
            input_seq[0] = input_vector;
            
            // Forward pass
            long double *output = SAM_forward(session->sam_model, input_seq, 1);
            
            if (output) {
                // Calculate loss (simplified MSE)
                double loss = 0.0;
                for (int j = 0; j < CONTEXT_DIM; j++) {
                    double diff = (double)output[j] - (double)target_vector[j];
                    loss += diff * diff;
                }
                loss /= CONTEXT_DIM;
                
                samples[i].loss = loss;
                total_loss += loss;
                trained_samples++;
                
                printf("    ✅ Loss: %.6f\n", loss);
                
                // Simple backpropagation (simplified)
                SAM_backprop(session->sam_model, input_seq, 1, target_vector);
                
                free(output);
            } else {
                printf("    ❌ Forward pass failed\n");
            }
            
            // Cleanup
            free(input_seq);
            free(input_vector);
            free(target_vector);
            
            samples[i].processed = 1;
            
            // Small delay to prevent overwhelming
            usleep(100000); // 0.1 second
        }
    }
    
    if (trained_samples > 0) {
        session->average_loss = total_loss / trained_samples;
        session->total_samples += trained_samples;
        
        printf("📊 Training completed:\n");
        printf("  Trained samples: %d\n", trained_samples);
        printf("  Average loss: %.6f\n", session->average_loss);
        printf("  Total samples: %d\n", session->total_samples);
        
        // Log training session
        if (session->log_file) {
            time_t now = time(NULL);
            fprintf(session->log_file, "[%ld] Epoch %d: Trained %d samples, Loss: %.6f\n", 
                    now, session->epoch_count, trained_samples, session->average_loss);
            fflush(session->log_file);
        }
    } else {
        printf("⚠️  No samples were trained\n");
    }
}

// Save model checkpoint
void save_model_checkpoint(TrainingSession *session) {
    printf("💾 Saving model checkpoint...\n");
    
    char checkpoint_name[100];
    snprintf(checkpoint_name, sizeof(checkpoint_name), "continuous_training_epoch_%d.bin", session->epoch_count);
    
    if (SAM_save(session->sam_model, checkpoint_name)) {
        printf("✅ Checkpoint saved: %s\n", checkpoint_name);
    } else {
        printf("❌ Failed to save checkpoint\n");
    }
}

// Display training status
void display_training_status(TrainingSession *session) {
    time_t now = time(NULL);
    int elapsed = now - session->session_start;
    int hours = elapsed / 3600;
    int minutes = (elapsed % 3600) / 60;
    int seconds = elapsed % 60;
    
    printf("\n" "=" * 60 "\n");
    printf("🎓 CONTINUOUS TRAINING STATUS\n");
    printf("=" * 60 "\n");
    printf("Session Time: %02d:%02d:%02d\n", hours, minutes, seconds);
    printf("Epoch: %d\n", session->epoch_count);
    printf("Total Samples: %d\n", session->total_samples);
    printf("Average Loss: %.6f\n", session->average_loss);
    printf("Ollama Model: %s\n", session->ollama_model);
    printf("Status: %s\n", keep_running ? "🟢 Running" : "🔴 Stopping");
    printf("=" * 60 "\n\n");
}

// Main continuous training loop
void continuous_training_loop(TrainingSession *session) {
    TrainingSample samples[MAX_TRAINING_SAMPLES];
    int sample_count = 0;
    
    printf("🚀 Starting continuous training loop...\n");
    printf("💡 Press Ctrl+C to stop gracefully\n\n");
    
    while (keep_running) {
        time_t now = time(NULL);
        
        // Check if it's time to train
        if (now - session->last_training >= TRAINING_INTERVAL) {
            printf("⏰ Training interval reached (%d seconds)\n", TRAINING_INTERVAL);
            
            // Generate new training samples
            generate_training_samples(session, samples, &sample_count);
            
            // Train the model
            if (sample_count > 0) {
                train_sam_model(session, samples, sample_count);
                session->epoch_count++;
                session->last_training = now;
                
                // Save checkpoint every 5 epochs
                if (session->epoch_count % 5 == 0) {
                    save_model_checkpoint(session);
                }
                
                // Display status
                display_training_status(session);
            }
            
            // Reset for next iteration
            sample_count = 0;
        }
        
        // Sleep for a short time
        sleep(1);
    }
}

// Initialize training session
int init_training_session(TrainingSession *session, const char *ollama_model) {
    memset(session, 0, sizeof(TrainingSession));
    
    // Initialize SAM model
    printf("🤖 Initializing SAM model...\n");
    session->sam_model = SAM_load("stage4_response_final.bin");
    if (!session->sam_model) {
        printf("❌ Failed to load SAM model\n");
        return 0;
    }
    printf("✅ SAM model loaded\n");
    
    // Set Ollama model
    strncpy(session->ollama_model, ollama_model, sizeof(session->ollama_model) - 1);
    
    // Initialize session
    session->running = 1;
    session->epoch_count = 0;
    session->total_samples = 0;
    session->average_loss = 0.0;
    session->session_start = time(NULL);
    session->last_training = 0;
    
    // Open log file
    char log_filename[100];
    time_t now = time(NULL);
    snprintf(log_filename, sizeof(log_filename), "continuous_training_%ld.log", now);
    session->log_file = fopen(log_filename, "w");
    
    if (session->log_file) {
        fprintf(session->log_file, "Continuous training session started at %ld\n", now);
        fprintf(session->log_file, "Ollama model: %s\n", session->ollama_model);
        fflush(session->log_file);
        printf("📝 Log file: %s\n", log_filename);
    } else {
        printf("⚠️  Could not open log file\n");
    }
    
    return 1;
}

// Cleanup training session
void cleanup_training_session(TrainingSession *session) {
    printf("🧹 Cleaning up training session...\n");
    
    if (session->sam_model) {
        SAM_destroy(session->sam_model);
    }
    
    if (session->log_file) {
        fprintf(session->log_file, "Training session ended at %ld\n", time(NULL));
        fclose(session->log_file);
    }
    
    // Save final checkpoint
    if (session->epoch_count > 0) {
        save_model_checkpoint(session);
    }
    
    printf("✅ Cleanup completed\n");
}

int main(int argc, char *argv[]) {
    printf("=== CONTINUOUS TRAINING WITH OLLAMA ===\n");
    printf("Using Ollama to generate training data for SAM model\n");
    printf("=========================================\n\n");
    
    // Parse command line arguments
    char ollama_model[100] = OLLAMA_MODEL;
    
    if (argc > 1) {
        strncpy(ollama_model, argv[1], sizeof(ollama_model) - 1);
        printf("Using Ollama model: %s\n", ollama_model);
    } else {
        printf("Using default Ollama model: %s\n", ollama_model);
    }
    
    // Initialize signal handlers
    init_signal_handlers();
    
    // Check Ollama availability
    if (!check_ollama_available()) {
        printf("❌ Cannot proceed without Ollama\n");
        return 1;
    }
    
    // Initialize training session
    TrainingSession session;
    if (!init_training_session(&session, ollama_model)) {
        printf("❌ Failed to initialize training session\n");
        return 1;
    }
    
    // Display initial status
    display_training_status(&session);
    
    // Start continuous training
    continuous_training_loop(&session);
    
    // Cleanup
    cleanup_training_session(&session);
    
    printf("\n🎉 Continuous training session completed\n");
    printf("📊 Final statistics:\n");
    printf("  Total epochs: %d\n", session.epoch_count);
    printf("  Total samples: %d\n", session.total_samples);
    printf("  Final average loss: %.6f\n", session.average_loss);
    
    return 0;
}
