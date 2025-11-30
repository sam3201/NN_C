#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../SAM/SAM.h"

#define MAX_RESPONSE_LENGTH 4096

// Structure for HF communication result
typedef struct {
    char* prompt;
    char* response;
    char* model_name;
} HFResponse;

// Send prompt to Hugging Face model and get response
int hf_query(const char* model_name, const char* prompt, char* response, size_t response_size) {
    char command[2048];
    snprintf(command, sizeof(command),
             "python3 hf_communicator.py '%s' '%s'", model_name, prompt);
    
    FILE* pipe = popen(command, "r");
    if (!pipe) {
        fprintf(stderr, "Error running HF communicator\n");
        return 0;
    }
    
    // Read response from JSON file
    FILE* json_file = fopen("hf_response.json", "r");
    if (!json_file) {
        pclose(pipe);
        return 0;
    }
    
    // Simple JSON parsing (extract response field)
    char line[4096];
    int in_response = 0;
    size_t response_pos = 0;
    
    while (fgets(line, sizeof(line), json_file)) {
        if (strstr(line, "\"response\"")) {
            in_response = 1;
            // Extract response value
            char* start = strchr(line, ':');
            if (start) {
                start++; // Skip ':'
                while (*start == ' ' || *start == '"') start++;
                char* end = strrchr(start, '"');
                if (end) {
                    size_t len = end - start;
                    if (len < response_size - 1) {
                        strncpy(response, start, len);
                        response[len] = '\0';
                    }
                }
            }
        }
    }
    
    fclose(json_file);
    pclose(pipe);
    
    return strlen(response) > 0;
}

// Load prompt from file or use default
int load_prompt(const char* prompt_file, char* prompt, size_t prompt_size) {
    FILE* f = fopen(prompt_file, "r");
    if (!f) {
        return 0;  // File doesn't exist, use default
    }
    
    // Read first line as prompt
    if (fgets(prompt, prompt_size, f)) {
        // Remove newline
        prompt[strcspn(prompt, "\n")] = 0;
        fclose(f);
        return 1;
    }
    
    fclose(f);
    return 0;
}

// SAM communicates with HF model - continues conversation until user stops
void sam_hf_dialogue(SAM_t* sam, const char* hf_model_name, const char* initial_prompt) {
    printf("=== SAM-Hugging Face Continuous Dialogue ===\n\n");
    
    char current_prompt[2048];
    char hf_response[MAX_RESPONSE_LENGTH];
    char sam_response[2048];
    int conversation_round = 0;
    
    // Start with initial prompt
    strncpy(current_prompt, initial_prompt, sizeof(current_prompt) - 1);
    current_prompt[sizeof(current_prompt) - 1] = '\0';
    
    printf("Starting conversation with prompt:\n");
    printf("----------------------------------------\n");
    printf("%s\n", current_prompt);
    printf("----------------------------------------\n\n");
    printf("(Conversation will continue. Type 'stop' or 'quit' to end, 'new' for new prompt)\n\n");
    
    while (1) {
        conversation_round++;
        printf("\n[Round %d]\n", conversation_round);
        printf("SAM Query: %s\n\n", current_prompt);
        
        // Query HF model
        printf("Querying %s...\n", hf_model_name);
        if (!hf_query(hf_model_name, current_prompt, hf_response, sizeof(hf_response))) {
            printf("Error: Failed to get response from HF model\n");
            break;
        }
        
        printf("HF Model Response:\n");
        printf("----------------------------------------\n");
        printf("%s\n", hf_response);
        printf("----------------------------------------\n\n");
        
        // SAM processes the response
        printf("SAM Processing Response...\n");
        
        size_t input_dim = sam->layer_sizes[0];
        long double* input = (long double*)calloc(input_dim, sizeof(long double));
        
        size_t response_len = strlen(hf_response);
        size_t copy_len = (response_len < input_dim) ? response_len : input_dim;
        
        for (size_t i = 0; i < copy_len; i++) {
            input[i] = ((long double)((unsigned char)hf_response[i])) / 255.0L;
        }
        
        long double** input_seq = (long double**)malloc(sizeof(long double*));
        input_seq[0] = input;
        
        // SAM forward pass
        long double* sam_output = SAM_forward(sam, input_seq, 1);
        
        if (sam_output) {
            // Decode SAM's response
            size_t output_dim = sam->layer_sizes[sam->num_layers - 1];
            size_t decode_len = (output_dim < sizeof(sam_response) - 1) ? output_dim : sizeof(sam_response) - 1;
            
            for (size_t i = 0; i < decode_len; i++) {
                int ascii = (int)(sam_output[i] * 255.0L);
                if (ascii >= 32 && ascii <= 126) {
                    sam_response[i] = (char)ascii;
                } else {
                    sam_response[i] = ' ';
                }
            }
            sam_response[decode_len] = '\0';
            
            printf("SAM Internal Response: %s\n", sam_response);
            free(sam_output);
        }
        
        // SAM adapts based on HF response
        SAM_adapt(sam, input_seq, 1);
        
        free(input);
        free(input_seq);
        
        // Continue conversation - use HF response as next prompt, or ask user
        printf("\nOptions:\n");
        printf("  [Enter] - Continue conversation (use HF response as next prompt)\n");
        printf("  'new' - Enter new prompt\n");
        printf("  'stop' or 'quit' - End conversation\n");
        printf("Choice: ");
        
        char choice[256];
        if (!fgets(choice, sizeof(choice), stdin)) {
            break;
        }
        
        // Remove newline
        choice[strcspn(choice, "\n")] = 0;
        
        if (strcmp(choice, "stop") == 0 || strcmp(choice, "quit") == 0 || strcmp(choice, "q") == 0) {
            printf("\nEnding conversation.\n");
            break;
        } else if (strcmp(choice, "new") == 0 || strcmp(choice, "n") == 0) {
            printf("Enter new prompt: ");
            if (fgets(current_prompt, sizeof(current_prompt), stdin)) {
                current_prompt[strcspn(current_prompt, "\n")] = 0;
            }
        } else {
            // Continue with HF response as next prompt (take first part)
            size_t prompt_len = strlen(hf_response);
            size_t take_len = (prompt_len < sizeof(current_prompt) - 1) ? prompt_len : sizeof(current_prompt) - 1;
            strncpy(current_prompt, hf_response, take_len);
            current_prompt[take_len] = '\0';
            
            // If response is too long, take first sentence or first 200 chars
            if (take_len > 200) {
                char* period = strchr(current_prompt, '.');
                if (period) {
                    *(period + 1) = '\0';
                } else {
                    current_prompt[200] = '\0';
                }
            }
        }
    }
    
    printf("\nSAM has processed %d rounds of conversation\n", conversation_round);
}

// Interactive dialogue between SAM and HF
void sam_hf_interactive(SAM_t* sam, const char* hf_model_name) {
    printf("=== SAM-Hugging Face Interactive Dialogue ===\n\n");
    printf("Type 'quit', 'stop', or 'q' to exit\n");
    printf("Type 'new' to start a new conversation thread\n\n");
    
    char user_input[1024];
    char hf_response[MAX_RESPONSE_LENGTH];
    int conversation_count = 0;
    
    while (1) {
        printf("You: ");
        if (!fgets(user_input, sizeof(user_input), stdin)) {
            break;
        }
        
        // Remove newline
        user_input[strcspn(user_input, "\n")] = 0;
        
        if (strcmp(user_input, "quit") == 0 || strcmp(user_input, "stop") == 0 || 
            strcmp(user_input, "q") == 0) {
            printf("\nEnding dialogue.\n");
            break;
        }
        
        if (strcmp(user_input, "new") == 0 || strcmp(user_input, "n") == 0) {
            printf("Starting new conversation thread...\n\n");
            conversation_count = 0;
            continue;
        }
        
        if (strlen(user_input) == 0) {
            continue;
        }
        
        conversation_count++;
        printf("\n[Exchange %d]\n", conversation_count);
        
        // Query HF model
        printf("Querying %s...\n", hf_model_name);
        if (hf_query(hf_model_name, user_input, hf_response, sizeof(hf_response))) {
            printf("HF: %s\n\n", hf_response);
            
            // SAM processes the response
            printf("SAM processing response...\n");
            size_t input_dim = sam->layer_sizes[0];
            long double* input = (long double*)calloc(input_dim, sizeof(long double));
            
            size_t response_len = strlen(hf_response);
            size_t copy_len = (response_len < input_dim) ? response_len : input_dim;
            
            for (size_t i = 0; i < copy_len; i++) {
                input[i] = ((long double)((unsigned char)hf_response[i])) / 255.0L;
            }
            
            long double** input_seq = (long double**)malloc(sizeof(long double*));
            input_seq[0] = input;
            
            SAM_adapt(sam, input_seq, 1);
            
            printf("SAM has learned from this exchange.\n\n");
            
            free(input);
            free(input_seq);
        } else {
            printf("Error: Failed to get response from HF model\n\n");
        }
    }
    
    printf("\nDialogue ended after %d exchanges.\n", conversation_count);
}

int main(int argc, char* argv[]) {
    const char* hf_model = (argc > 1) ? argv[1] : "gpt2";
    int interactive = (argc > 2 && strcmp(argv[2], "interactive") == 0);
    const char* prompt_file = (argc > 3) ? argv[3] : "prompt.txt";
    const char* custom_prompt = NULL;
    
    // Check for custom prompt as argument
    if (argc > 2 && argv[2][0] != 'i' && strstr(argv[2], ".txt") == NULL) {
        // Might be a custom prompt
        if (strlen(argv[2]) > 10) {  // Likely a prompt, not a mode
            custom_prompt = argv[2];
        }
    }
    
    printf("=== SAM-Hugging Face Communication Bridge ===\n\n");
    printf("HF Model: %s\n", hf_model);
    printf("Mode: %s\n", interactive ? "Interactive" : "Continuous Dialogue");
    
    // Load or get initial prompt
    char initial_prompt[2048];
    int prompt_loaded = 0;
    
    if (custom_prompt) {
        strncpy(initial_prompt, custom_prompt, sizeof(initial_prompt) - 1);
        initial_prompt[sizeof(initial_prompt) - 1] = '\0';
        prompt_loaded = 1;
        printf("Using custom prompt from command line\n");
    } else if (load_prompt(prompt_file, initial_prompt, sizeof(initial_prompt))) {
        prompt_loaded = 1;
        printf("Loaded prompt from: %s\n", prompt_file);
    } else {
        // Default prompt
        strcpy(initial_prompt, "How can we create a model that self actualizes?");
        printf("Using default prompt\n");
        printf("(To use custom prompt: edit prompt.txt or provide as argument)\n");
    }
    
    printf("\n");
    
    // Initialize SAM model
    printf("Initializing SAM model...\n");
    SAM_t* sam = SAM_init(768, 768, 8, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM model\n");
        return 1;
    }
    
    // Try to load existing model
    SAM_t* loaded_sam = SAM_load("../sam_trained_model.bin");
    if (loaded_sam) {
        SAM_destroy(sam);
        sam = loaded_sam;
        printf("Loaded existing SAM model\n");
    } else {
        printf("Using new SAM model\n");
    }
    
    printf("\n");
    
    if (interactive) {
        sam_hf_interactive(sam, hf_model);
    } else {
        // Continuous dialogue mode
        sam_hf_dialogue(sam, hf_model, initial_prompt);
    }
    
    // Save SAM model after dialogue
    printf("\nSaving SAM model...\n");
    char timestamp[64];
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    info = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", info);
    
    char filename[128];
    snprintf(filename, sizeof(filename), "../sam_hf_dialogue_%s.bin", timestamp);
    
    if (SAM_save(sam, filename) == 1) {
        printf("✓ Model saved to %s\n", filename);
    }
    
    SAM_destroy(sam);
    
    printf("\n=== Communication Complete ===\n");
    return 0;
}

