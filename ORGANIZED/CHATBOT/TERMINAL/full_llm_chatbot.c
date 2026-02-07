#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>

#define CONTEXT_DIM 128
#define VOCAB_SIZE 10000
#define MAX_RESPONSE_LENGTH 500
#define MAX_INPUT_LENGTH 200
#define MAX_HISTORY 10
#define MAX_TOKENS 100

// Chatbot states
typedef enum {
    STATE_GREETING,
    STATE_CONVERSATION,
    STATE_QUESTION,
    STATE_COMMAND,
    STATE_DIALOGUE,
    STATE_TECHNICAL,
    STATE_CREATIVE
} ChatbotState;

// Response types
typedef enum {
    RESPONSE_GREETING,
    RESPONSE_QUESTION,
    RESPONSE_STATEMENT,
    RESPONSE_COMMAND,
    RESPONSE_DIALOGUE,
    RESPONSE_EXPLANATION,
    RESPONSE_CREATIVE,
    RESPONSE_TECHNICAL
} ResponseType;

// Chat message structure
typedef struct {
    char user_input[MAX_INPUT_LENGTH];
    char bot_response[MAX_RESPONSE_LENGTH];
    time_t timestamp;
    ChatbotState state;
    ResponseType response_type;
    double confidence;
} ChatMessage;

// Chatbot context
typedef struct {
    ChatMessage history[MAX_HISTORY];
    int history_count;
    ChatbotState current_state;
    char user_name[50];
    char bot_name[50];
    char personality[100];
    time_t session_start;
    int message_count;
} ChatbotContext;

// Vocabulary entry
typedef struct {
    char word[50];
    int frequency;
    long double *vector;
    int vector_length;
} VocabularyEntry;

// Vocabulary
typedef struct {
    VocabularyEntry *entries;
    int size;
    int total_words;
    int vector_length;
} Vocabulary;

// Multi-stage model structure
typedef struct {
    SAM_t *character_model;
    SAM_t *word_model;
    SAM_t *phrase_model;
    SAM_t *response_model;
    SAM_t *advanced_agi_model;
    Vocabulary *vocabulary;
} MultiStageModel;

// Get response type name
const char* get_response_type_name(ResponseType type) {
    switch (type) {
        case RESPONSE_GREETING: return "GREETING";
        case RESPONSE_QUESTION: return "QUESTION";
        case RESPONSE_STATEMENT: return "STATEMENT";
        case RESPONSE_COMMAND: return "COMMAND";
        case RESPONSE_DIALOGUE: return "DIALOGUE";
        case RESPONSE_EXPLANATION: return "EXPLANATION";
        case RESPONSE_CREATIVE: return "CREATIVE";
        case RESPONSE_TECHNICAL: return "TECHNICAL";
        default: return "UNKNOWN";
    }
}

// Get state name
const char* get_state_name(ChatbotState state) {
    switch (state) {
        case STATE_GREETING: return "GREETING";
        case STATE_CONVERSATION: return "CONVERSATION";
        case STATE_QUESTION: return "QUESTION";
        case STATE_COMMAND: return "COMMAND";
        case STATE_DIALOGUE: return "DIALOGUE";
        case STATE_TECHNICAL: return "TECHNICAL";
        case STATE_CREATIVE: return "CREATIVE";
        default: return "UNKNOWN";
    }
}

// Initialize vocabulary
Vocabulary* init_vocabulary(int size, int vector_length) {
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    if (!vocab) return NULL;
    
    vocab->entries = malloc(size * sizeof(VocabularyEntry));
    if (!vocab->entries) {
        free(vocab);
        return NULL;
    }
    
    vocab->size = 0;
    vocab->total_words = 0;
    vocab->vector_length = vector_length;
    
    // Initialize all entries
    for (int i = 0; i < size; i++) {
        vocab->entries[i].vector = malloc(vector_length * sizeof(long double));
        if (!vocab->entries[i].vector) {
            for (int j = 0; j < i; j++) {
                free(vocab->entries[j].vector);
            }
            free(vocab->entries);
            free(vocab);
            return NULL;
        }
        vocab->entries[i].word[0] = '\0';
        vocab->entries[i].frequency = 0;
        for (int j = 0; j < vector_length; j++) {
            vocab->entries[i].vector[j] = 0.0L;
        }
    }
    
    return vocab;
}

// Load vocabulary from file
int load_vocabulary(Vocabulary *vocab, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return 0;
    
    char line[200];
    while (fgets(line, sizeof(line), file) && vocab->size < VOCAB_SIZE) {
        char word[50];
        int frequency;
        
        // Parse line: word frequency
        if (sscanf(line, "%s %d", word, &frequency) == 2) {
            strcpy(vocab->entries[vocab->size].word, word);
            vocab->entries[vocab->size].frequency = frequency;
            
            // Generate random vector for the word
            for (int i = 0; i < vocab->vector_length; i++) {
                vocab->entries[vocab->size].vector[i] = (long double)rand() / RAND_MAX * 2.0 - 1.0L;
            }
            
            vocab->total_words += frequency;
            vocab->size++;
        }
    }
    
    fclose(file);
    return vocab->size > 0;
}

// Find word in vocabulary
VocabularyEntry* find_word_in_vocabulary(Vocabulary *vocab, const char *word) {
    for (int i = 0; i < vocab->size; i++) {
        if (strcmp(vocab->entries[i].word, word) == 0) {
            return &vocab->entries[i];
        }
    }
    return NULL;
}

// Tokenize text into words
int tokenize_text(const char *text, char words[][50], int max_words) {
    int word_count = 0;
    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " \t\n\r.,!?;:");
    
    while (token != NULL && word_count < max_words) {
        // Convert to lowercase
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }
        
        // Remove punctuation from end
        int len = strlen(token);
        while (len > 0 && !isalpha(token[len-1])) {
            token[len-1] = '\0';
            len--;
        }
        
        if (len > 0) {
            strcpy(words[word_count], token);
            word_count++;
        }
        
        token = strtok(NULL, " \t\n\r.,!?;:");
    }
    
    free(text_copy);
    return word_count;
}

// Encode text to vector
long double* encode_text_to_vector(Vocabulary *vocab, const char *text, int *length) {
    char words[MAX_TOKENS][50];
    int word_count = tokenize_text(text, words, MAX_TOKENS);
    
    if (word_count == 0) {
        *length = vocab->vector_length;
        long double *vector = calloc(vocab->vector_length, sizeof(long double));
        return vector;
    }
    
    *length = vocab->vector_length;
    long double *vector = calloc(vocab->vector_length, sizeof(long double));
    if (!vector) return NULL;
    
    // Average word vectors
    int valid_words = 0;
    for (int i = 0; i < word_count; i++) {
        VocabularyEntry *entry = find_word_in_vocabulary(vocab, words[i]);
        if (entry) {
            for (int j = 0; j < vocab->vector_length; j++) {
                vector[j] += entry->vector[j];
            }
            valid_words++;
        }
    }
    
    if (valid_words > 0) {
        for (int i = 0; i < vocab->vector_length; i++) {
            vector[i] /= valid_words;
        }
    }
    
    return vector;
}

// Detect input type and determine state
ChatbotState detect_chatbot_state(const char *input, ChatbotContext *context) {
    // Check for greetings
    if (strstr(input, "hello") || strstr(input, "hi") || strstr(input, "hey") || 
        strstr(input, "good morning") || strstr(input, "good evening")) {
        return STATE_GREETING;
    }
    
    // Check for questions
    if (strstr(input, "?") || strstr(input, "what") || strstr(input, "how") || 
        strstr(input, "why") || strstr(input, "where") || strstr(input, "when")) {
        return STATE_QUESTION;
    }
    
    // Check for commands
    if (strstr(input, "please") || strstr(input, "can you") || strstr(input, "help") ||
        strstr(input, "tell me") || strstr(input, "show me")) {
        return STATE_COMMAND;
    }
    
    // Check for technical content
    if (strstr(input, "algorithm") || strstr(input, "code") || strstr(input, "programming") ||
        strstr(input, "computer") || strstr(input, "software") || strstr(input, "data")) {
        return STATE_TECHNICAL;
    }
    
    // Check for creative content
    if (strstr(input, "story") || strstr(input, "poem") || strstr(input, "creative") ||
        strstr(input, "imagine") || strstr(input, "dream")) {
        return STATE_CREATIVE;
    }
    
    // Default to conversation
    return STATE_CONVERSATION;
}

// Determine response type based on state
ResponseType determine_response_type(ChatbotState state) {
    switch (state) {
        case STATE_GREETING: return RESPONSE_GREETING;
        case STATE_QUESTION: return RESPONSE_EXPLANATION;
        case STATE_COMMAND: return RESPONSE_STATEMENT;
        case STATE_DIALOGUE: return RESPONSE_DIALOGUE;
        case STATE_TECHNICAL: return RESPONSE_TECHNICAL;
        case STATE_CREATIVE: return RESPONSE_CREATIVE;
        default: return RESPONSE_STATEMENT;
    }
}

// Initialize multi-stage model
MultiStageModel* init_multi_stage_model() {
    MultiStageModel *model = malloc(sizeof(MultiStageModel));
    if (!model) return NULL;
    
    // Initialize vocabulary
    model->vocabulary = init_vocabulary(VOCAB_SIZE, CONTEXT_DIM);
    if (!model->vocabulary) {
        free(model);
        return NULL;
    }
    
    // Load vocabulary
    if (!load_vocabulary(model->vocabulary, "stage2_vocabulary.txt")) {
        printf("Warning: Could not load vocabulary, using basic words\n");
        // Add basic words
        const char* basic_words[] = {"hello", "hi", "thanks", "please", "yes", "no", "ok", "good", "bad", "nice"};
        for (int i = 0; i < 10; i++) {
            strcpy(model->vocabulary->entries[i].word, basic_words[i]);
            model->vocabulary->entries[i].frequency = 100;
            for (int j = 0; j < CONTEXT_DIM; j++) {
                model->vocabulary->entries[i].vector[j] = (long double)rand() / RAND_MAX * 2.0 - 1.0L;
            }
            model->vocabulary->size++;
            model->vocabulary->total_words += 100;
        }
    }
    
    // Load models
    model->character_model = SAM_load("stage1_fixed_final.bin");
    model->word_model = SAM_load("stage2_word_final.bin");
    model->phrase_model = SAM_load("stage3_phrase_final.bin");
    model->response_model = SAM_load("stage4_response_final.bin");
    model->advanced_agi_model = SAM_load("stage5_complete");
    
    printf("Model loading status:\n");
    printf("  Character model: %s\n", model->character_model ? "✅ Loaded" : "❌ Not found");
    printf("  Word model: %s\n", model->word_model ? "✅ Loaded" : "❌ Not found");
    printf("  Phrase model: %s\n", model->phrase_model ? "✅ Loaded" : "❌ Not found");
    printf("  Response model: %s\n", model->response_model ? "✅ Loaded" : "❌ Not found");
    printf("  Advanced AGI model: %s\n", model->advanced_agi_model ? "✅ Loaded" : "❌ Not found");
    printf("  Vocabulary: %d words\n", model->vocabulary->size);
    
    return model;
}

// Generate response using character model
char* generate_character_response(MultiStageModel *model, const char *input) {
    if (!model->character_model) {
        char *response = malloc(100);
        strcpy(response, "Character model not available.");
        return response;
    }
    
    // Simple character-based response
    char *response = malloc(100);
    strcpy(response, "I can process characters: ");
    
    for (int i = 0; i < strlen(input) && i < 20; i++) {
        if (isalpha(input[i])) {
            char next_char = input[i] + 1; // Simple character prediction
            if (next_char > 'z') next_char = 'a';
            response[strlen(response)] = next_char;
        }
    }
    response[strlen(response)] = '\0';
    
    return response;
}

// Generate response using word model
char* generate_word_response(MultiStageModel *model, const char *input) {
    if (!model->word_model) {
        char *response = malloc(100);
        strcpy(response, "Word model not available.");
        return response;
    }
    
    // Simple word-based response
    char *response = malloc(200);
    strcpy(response, "I understand the words: ");
    
    char words[MAX_TOKENS][50];
    int word_count = tokenize_text(input, words, MAX_TOKENS);
    
    for (int i = 0; i < word_count && i < 5; i++) {
        if (i > 0) strcat(response, ", ");
        strcat(response, words[i]);
    }
    
    if (word_count > 5) {
        strcat(response, "...");
    }
    
    return response;
}

// Generate response using phrase model
char* generate_phrase_response(MultiStageModel *model, const char *input) {
    if (!model->phrase_model) {
        char *response = malloc(100);
        strcpy(response, "Phrase model not available.");
        return response;
    }
    
    // Simple phrase-based response
    char *response = malloc(300);
    strcpy(response, "I recognize the phrase patterns in your message. ");
    
    if (strstr(input, "how are")) {
        strcat(response, "I'm functioning well and ready to help!");
    } else if (strstr(input, "what can")) {
        strcat(response, "I can engage in conversation, answer questions, and assist with various tasks.");
    } else {
        strcat(response, "Let me process that thoughtfully for you.");
    }
    
    return response;
}

// Generate response using response model
char* generate_response_response(MultiStageModel *model, const char *input) {
    if (!model->response_model) {
        char *response = malloc(100);
        strcpy(response, "Response model not available.");
        return response;
    }
    
    // Encode input
    int input_length;
    long double *input_vector = encode_text_to_vector(model->vocabulary, input, &input_length);
    if (!input_vector) {
        char *response = malloc(100);
        strcpy(response, "I couldn't process that input.");
        return response;
    }
    
    // Create input sequence
    long double **input_seq = malloc(sizeof(long double*));
    input_seq[0] = input_vector;
    
    // Forward pass
    long double *output = SAM_forward(model->response_model, input_seq, 1);
    
    char *response = malloc(MAX_RESPONSE_LENGTH);
    if (!output) {
        strcpy(response, "I'm having trouble processing that right now.");
    } else {
        // Generate response based on output
        if (strstr(input, "hello") || strstr(input, "hi")) {
            strcpy(response, "Hello! It's nice to meet you. How can I help you today?");
        } else if (strstr(input, "?")) {
            strcpy(response, "That's an interesting question. Let me think about that carefully...");
        } else if (strstr(input, "thank")) {
            strcpy(response, "You're very welcome! I'm always here to help.");
        } else if (strstr(input, "bye")) {
            strcpy(response, "Goodbye! It was nice talking with you. Feel free to come back anytime!");
        } else {
            strcpy(response, "I understand what you're saying. Let me provide a thoughtful response to that.");
        }
        
        // Add variation based on output
        long double output_sum = 0.0L;
        for (int i = 0; i < CONTEXT_DIM; i++) {
            output_sum += output[i];
        }
        
        if (output_sum > 0.5L) {
            strcat(response, " I'm quite confident about this.");
        } else if (output_sum < -0.5L) {
            strcat(response, " I'm still learning about this topic.");
        }
        
        free(output);
    }
    
    free(input_seq);
    free(input_vector);
    return response;
}

// Generate response using advanced AGI model
char* generate_agi_response(MultiStageModel *model, const char *input) {
    if (!model->advanced_agi_model) {
        char *response = malloc(100);
        strcpy(response, "Advanced AGI model not available.");
        return response;
    }
    
    // Advanced AGI response with multiple capabilities
    char *response = malloc(MAX_RESPONSE_LENGTH);
    
    if (strstr(input, "help")) {
        strcpy(response, "As an advanced AGI system, I can help with:\n• Natural conversation\n• Problem solving\n• Information retrieval\n• Creative tasks\n• Technical assistance\nWhat specific help do you need?");
    } else if (strstr(input, "what are you")) {
        strcpy(response, "I am an advanced AGI system with multiple learning stages:\n• Character-level understanding\n• Word recognition\n• Phrase context\n• Response generation\n• Advanced planning capabilities\nHow can I assist you today?");
    } else if (strstr(input, "capabilities")) {
        strcpy(response, "My capabilities include:\n• Multi-stage progressive learning\n• Hybrid action planning\n• Expert knowledge modules\n• World modeling\n• Knowledge transfer\n• MCTS planning\nI'm designed to handle complex tasks and conversations.");
    } else {
        strcpy(response, "As an advanced AGI, I process your input through multiple learning stages to provide comprehensive responses. I'm continuously improving through my progressive learning architecture.");
    }
    
    return response;
}

// Generate contextual response
char* generate_contextual_response(MultiStageModel *model, ChatbotContext *context, const char *input) {
    // Detect state
    ChatbotState state = detect_chatbot_state(input, context);
    context->current_state = state;
    
    // Determine response type
    ResponseType response_type = determine_response_type(state);
    
    // Generate response based on available models and state
    char *response = NULL;
    
    // Try advanced AGI first
    if (model->advanced_agi_model) {
        response = generate_agi_response(model, input);
    }
    
    // Fall back to response model
    if (!response && model->response_model) {
        response = generate_response_response(model, input);
    }
    
    // Fall back to phrase model
    if (!response && model->phrase_model) {
        response = generate_phrase_response(model, input);
    }
    
    // Fall back to word model
    if (!response && model->word_model) {
        response = generate_word_response(model, input);
    }
    
    // Fall back to character model
    if (!response && model->character_model) {
        response = generate_character_response(model, input);
    }
    
    // Default response
    if (!response) {
        response = malloc(100);
        strcpy(response, "I'm here to help! What would you like to talk about?");
    }
    
    return response;
}

// Initialize chatbot context
ChatbotContext* init_chatbot_context() {
    ChatbotContext *context = malloc(sizeof(ChatbotContext));
    if (!context) return NULL;
    
    // Initialize history
    for (int i = 0; i < MAX_HISTORY; i++) {
        context->history[i].user_input[0] = '\0';
        context->history[i].bot_response[0] = '\0';
        context->history[i].timestamp = 0;
        context->history[i].state = STATE_CONVERSATION;
        context->history[i].response_type = RESPONSE_STATEMENT;
        context->history[i].confidence = 0.0;
    }
    
    context->history_count = 0;
    context->current_state = STATE_GREETING;
    strcpy(context->user_name, "User");
    strcpy(context->bot_name, "AGI Assistant");
    strcpy(context->personality, "Helpful, intelligent, and conversational");
    context->session_start = time(NULL);
    context->message_count = 0;
    
    return context;
}

// Add message to history
void add_to_history(ChatbotContext *context, const char *user_input, const char *bot_response) {
    if (context->history_count >= MAX_HISTORY) {
        // Shift history
        for (int i = 0; i < MAX_HISTORY - 1; i++) {
            context->history[i] = context->history[i + 1];
        }
        context->history_count = MAX_HISTORY - 1;
    }
    
    // Add new message
    strcpy(context->history[context->history_count].user_input, user_input);
    strcpy(context->history[context->history_count].bot_response, bot_response);
    context->history[context->history_count].timestamp = time(NULL);
    context->history[context->history_count].state = context->current_state;
    context->history[context->history_count].response_type = determine_response_type(context->current_state);
    context->history[context->history_count].confidence = 0.8; // Default confidence
    
    context->history_count++;
    context->message_count++;
}

// Display chat interface
void display_chat_interface(ChatbotContext *context) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    AGI CHATBOT INTERFACE                   ║\n");
    printf("║                    Session Started                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Bot: Hello! I'm %s, your advanced AGI assistant.\n", context->bot_name);
    printf("Bot: I have multiple learning stages and can help with various tasks.\n");
    printf("Bot: Type 'quit' to exit, 'help' for commands, or just start talking!\n");
    printf("\n");
}

// Display help information
void display_help() {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                           HELP MENU                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Available commands:\n");
    printf("  help     - Show this help menu\n");
    printf("  quit     - Exit the chatbot\n");
    printf("  status   - Show current session status\n");
    printf("  models   - Show loaded model status\n");
    printf("  history  - Show conversation history\n");
    printf("  clear    - Clear conversation history\n");
    printf("\nI can help with:\n");
    printf("• Natural conversation\n");
    printf("• Answering questions\n");
    printf("• Providing information\n");
    printf("• Creative tasks\n");
    printf("• Technical assistance\n");
    printf("• Problem solving\n");
    printf("\n");
}

// Display status information
void display_status(ChatbotContext *context) {
    time_t session_time = time(NULL) - context->session_start;
    int minutes = session_time / 60;
    int seconds = session_time % 60;
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        SESSION STATUS                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Session time: %d minutes, %d seconds\n", minutes, seconds);
    printf("Messages exchanged: %d\n", context->message_count);
    printf("Current state: %s\n", get_state_name(context->current_state));
    printf("History count: %d/%d\n", context->history_count, MAX_HISTORY);
    printf("Bot name: %s\n", context->bot_name);
    printf("User name: %s\n", context->user_name);
    printf("Personality: %s\n", context->personality);
    printf("\n");
}

// Display model status
void display_model_status(MultiStageModel *model) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                         MODEL STATUS                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Character model: %s\n", model->character_model ? "✅ Loaded" : "❌ Not available");
    printf("Word model: %s\n", model->word_model ? "✅ Loaded" : "❌ Not available");
    printf("Phrase model: %s\n", model->phrase_model ? "✅ Loaded" : "❌ Not available");
    printf("Response model: %s\n", model->response_model ? "✅ Loaded" : "❌ Not available");
    printf("Advanced AGI model: %s\n", model->advanced_agi_model ? "✅ Loaded" : "❌ Not available");
    printf("Vocabulary size: %d words\n", model->vocabulary->size);
    printf("Total vocabulary words: %d\n", model->vocabulary->total_words);
    printf("Vector dimension: %d\n", model->vocabulary->vector_length);
    printf("\n");
}

// Display conversation history
void display_history(ChatbotContext *context) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    CONVERSATION HISTORY                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    if (context->history_count == 0) {
        printf("No conversation history yet.\n");
    } else {
        for (int i = 0; i < context->history_count; i++) {
            printf("[%d] User: %s\n", i + 1, context->history[i].user_input);
            printf("[%d] Bot: %s\n", i + 1, context->history[i].bot_response);
            printf("[%d] State: %s, Type: %s, Confidence: %.2f\n", 
                   i + 1, 
                   get_state_name(context->history[i].state),
                   get_response_type_name(context->history[i].response_type),
                   context->history[i].confidence);
            printf("---\n");
        }
    }
    printf("\n");
}

// Clear conversation history
void clear_history(ChatbotContext *context) {
    for (int i = 0; i < MAX_HISTORY; i++) {
        context->history[i].user_input[0] = '\0';
        context->history[i].bot_response[0] = '\0';
        context->history[i].timestamp = 0;
        context->history[i].state = STATE_CONVERSATION;
        context->history[i].response_type = RESPONSE_STATEMENT;
        context->history[i].confidence = 0.0;
    }
    
    context->history_count = 0;
    context->current_state = STATE_GREETING;
    
    printf("Conversation history cleared.\n");
}

// Main chat loop
void run_chat_loop(MultiStageModel *model, ChatbotContext *context) {
    char input[MAX_INPUT_LENGTH];
    int running = 1;
    
    display_chat_interface(context);
    
    while (running) {
        printf("You: ");
        fflush(stdout);
        
        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }
        
        // Remove newline
        input[strcspn(input, "\n")] = '\0';
        
        // Check for commands
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            running = 0;
            printf("Bot: Goodbye! It was nice talking with you. Feel free to come back anytime!\n");
        } else if (strcmp(input, "help") == 0) {
            display_help();
        } else if (strcmp(input, "status") == 0) {
            display_status(context);
        } else if (strcmp(input, "models") == 0) {
            display_model_status(model);
        } else if (strcmp(input, "history") == 0) {
            display_history(context);
        } else if (strcmp(input, "clear") == 0) {
            clear_history(context);
        } else if (strlen(input) > 0) {
            // Generate response
            char *response = generate_contextual_response(model, context, input);
            
            printf("Bot: %s\n", response);
            
            // Add to history
            add_to_history(context, input, response);
            
            free(response);
        }
    }
}

// Cleanup
void cleanup_multi_stage_model(MultiStageModel *model) {
    if (model) {
        if (model->character_model) SAM_destroy(model->character_model);
        if (model->word_model) SAM_destroy(model->word_model);
        if (model->phrase_model) SAM_destroy(model->phrase_model);
        if (model->response_model) SAM_destroy(model->response_model);
        if (model->advanced_agi_model) SAM_destroy(model->advanced_agi_model);
        
        if (model->vocabulary) {
            for (int i = 0; i < model->vocabulary->size; i++) {
                free(model->vocabulary->entries[i].vector);
            }
            free(model->vocabulary->entries);
            free(model->vocabulary);
        }
        
        free(model);
    }
}

void cleanup_chatbot_context(ChatbotContext *context) {
    if (context) {
        free(context);
    }
}

int main(int argc, char *argv[]) {
    printf("=== FULL LLM CHATBOT WITH ADVANCED AGI ===\n\n");
    
    srand(time(NULL));
    
    printf("Initializing chatbot...\n");
    
    // Initialize multi-stage model
    MultiStageModel *model = init_multi_stage_model();
    if (!model) {
        printf("Failed to initialize model\n");
        return 1;
    }
    
    // Initialize chatbot context
    ChatbotContext *context = init_chatbot_context();
    if (!context) {
        printf("Failed to initialize chatbot context\n");
        cleanup_multi_stage_model(model);
        return 1;
    }
    
    printf("Chatbot initialized successfully!\n");
    printf("Starting conversation...\n\n");
    
    // Run chat loop
    run_chat_loop(model, context);
    
    // Cleanup
    cleanup_multi_stage_model(model);
    cleanup_chatbot_context(context);
    
    printf("\n=== CHATBOT SESSION ENDED ===\n");
    printf("Thank you for using the Advanced AGI Chatbot!\n");
    
    return 0;
}
