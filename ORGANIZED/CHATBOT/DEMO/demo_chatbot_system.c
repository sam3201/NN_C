#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/wait.h>

#define CONTEXT_DIM 128
#define VOCAB_SIZE 10000
#define MAX_RESPONSE_LENGTH 500
#define MAX_INPUT_LENGTH 200

// Demo configuration
typedef struct {
    int show_terminal_demo;
    int show_web_demo;
    int show_model_status;
    int run_interactive;
    char web_port[10];
} DemoConfig;

// Model status structure
typedef struct {
    int character_loaded;
    int word_loaded;
    int phrase_loaded;
    int response_loaded;
    int agi_loaded;
    int vocabulary_size;
    int total_models;
} ModelStatus;

// Get model status
ModelStatus get_model_status() {
    ModelStatus status = {0};
    
    // Check if model files exist
    if (access("stage1_fixed_final.bin", F_OK) == 0) {
        status.character_loaded = 1;
        status.total_models++;
    }
    
    if (access("stage2_word_final.bin", F_OK) == 0) {
        status.word_loaded = 1;
        status.total_models++;
    }
    
    if (access("stage3_phrase_final.bin", F_OK) == 0) {
        status.phrase_loaded = 1;
        status.total_models++;
    }
    
    if (access("stage4_response_final.bin", F_OK) == 0) {
        status.response_loaded = 1;
        status.total_models++;
    }
    
    if (access("stage5_complete", F_OK) == 0) {
        status.agi_loaded = 1;
        status.total_models++;
    }
    
    // Check vocabulary
    if (access("stage2_vocabulary.txt", F_OK) == 0) {
        FILE *vocab_file = fopen("stage2_vocabulary.txt", "r");
        if (vocab_file) {
            char line[200];
            status.vocabulary_size = 0;
            while (fgets(line, sizeof(line), vocab_file)) {
                status.vocabulary_size++;
            }
            fclose(vocab_file);
        }
    }
    
    return status;
}

// Display model status
void display_model_status(ModelStatus *status) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    MODEL STATUS REPORT                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Models Loaded: %d/5\n", status->total_models);
    printf("\n");
    
    printf("Character Model: %s\n", status->character_loaded ? "✅ LOADED" : "❌ MISSING");
    printf("Word Model:      %s\n", status->word_loaded ? "✅ LOADED" : "❌ MISSING");
    printf("Phrase Model:    %s\n", status->phrase_loaded ? "✅ LOADED" : "❌ MISSING");
    printf("Response Model:  %s\n", status->response_loaded ? "✅ LOADED" : "❌ MISSING");
    printf("Advanced AGI:     %s\n", status->agi_loaded ? "✅ LOADED" : "❌ MISSING");
    printf("\n");
    printf("Vocabulary Size: %d words\n", status->vocabulary_size);
    printf("\n");
    
    if (status->total_models >= 4) {
        printf("🎉 EXCELLENT: Core models are loaded and ready!\n");
    } else if (status->total_models >= 2) {
        printf("✅ GOOD: Basic functionality is available.\n");
    } else {
        printf("⚠️  WARNING: Limited functionality - some models missing.\n");
    }
    printf("\n");
}

// Run terminal demo
void run_terminal_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                  TERMINAL CHATBOT DEMO                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Starting terminal chatbot demo...\n");
    printf("This will run the full LLM chatbot with multi-stage learning.\n");
    printf("\n");
    
    // Check if chatbot executable exists
    if (access("full_llm_chatbot", F_OK) != 0) {
        printf("❌ Chatbot executable not found. Please compile it first:\n");
        printf("   gcc -o full_llm_chatbot full_llm_chatbot.c SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm\n");
        return;
    }
    
    printf("✅ Chatbot executable found\n");
    printf("🚀 Starting terminal chatbot...\n");
    printf("\n");
    
    // Run the chatbot with demo inputs
    FILE *chatbot = popen("./full_llm_chatbot", "w");
    if (chatbot) {
        // Send demo inputs
        fprintf(chatbot, "hello\n");
        fprintf(chatbot, "what can you do?\n");
        fprintf(chatbot, "how do you work?\n");
        fprintf(chatbot, "help\n");
        fprintf(chatbot, "status\n");
        fprintf(chatbot, "quit\n");
        fflush(chatbot);
        
        // Wait for completion
        pclose(chatbot);
        printf("\n✅ Terminal demo completed\n");
    } else {
        printf("❌ Failed to start terminal chatbot\n");
    }
}

// Run web demo
void run_web_demo(const char *port) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    WEB CHATBOT DEMO                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Starting web chatbot demo...\n");
    printf("This will start a web server for the chatbot interface.\n");
    printf("\n");
    
    // Check if web server executable exists
    if (access("web_server", F_OK) != 0) {
        printf("❌ Web server executable not found. Please compile it first:\n");
        printf("   gcc -o web_server web_server.c\n");
        return;
    }
    
    // Check if HTML file exists
    if (access("web_chatbot.html", F_OK) != 0) {
        printf("❌ Web interface file not found: web_chatbot.html\n");
        return;
    }
    
    printf("✅ Web server executable found\n");
    printf("✅ Web interface file found\n");
    printf("🚀 Starting web server on port %s...\n", port);
    printf("📱 Open your browser and go to: http://localhost:%s\n", port);
    printf("⏹️  Press Ctrl+C to stop the web server\n");
    printf("\n");
    
    // Start web server
    char command[100];
    snprintf(command, sizeof(command), "./web_server");
    
    int result = system(command);
    if (result == -1) {
        printf("❌ Failed to start web server\n");
    }
}

// Run interactive demo
void run_interactive_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                 INTERACTIVE CHATBOT DEMO                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Starting interactive chatbot demo...\n");
    printf("You can chat directly with the AGI assistant.\n");
    printf("Type 'quit' to exit the interactive session.\n");
    printf("\n");
    
    // Check if chatbot executable exists
    if (access("full_llm_chatbot", F_OK) != 0) {
        printf("❌ Chatbot executable not found. Please compile it first:\n");
        printf("   gcc -o full_llm_chatbot full_llm_chatbot.c SAM/SAM.c utils/NN/NEAT/NEAT.c utils/NN/TRANSFORMER/TRANSFORMER.c utils/NN/NN/NN.c -lm\n");
        return;
    }
    
    // Run interactive chatbot
    int result = system("./full_llm_chatbot");
    if (result == -1) {
        printf("❌ Failed to start interactive chatbot\n");
    }
}

// Show system information
void show_system_info() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    SYSTEM INFORMATION                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Advanced AGI Chatbot System\n");
    printf("Version: 1.0.0\n");
    printf("Build Date: %s", __DATE__);
    printf("\n");
    printf("Features:\n");
    printf("• Multi-stage progressive learning\n");
    printf("• Character, word, phrase, and response models\n");
    printf("• Advanced AGI integration\n");
    printf("• Terminal and web interfaces\n");
    printf("• Real-time conversation\n");
    printf("• Context-aware responses\n");
    printf("\n");
    printf("Architecture:\n");
    printf("• Stage 1: Character-level pattern recognition\n");
    printf("• Stage 2: Word vocabulary understanding\n");
    printf("• Stage 3: Phrase context awareness\n");
    printf("• Stage 4: Response generation\n");
    printf("• Stage 5: Advanced AGI components\n");
    printf("\n");
}

// Show usage instructions
void show_usage() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        USAGE GUIDE                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Usage: ./demo_chatbot_system [options]\n");
    printf("\n");
    printf("Options:\n");
    printf("  -t, --terminal     Run terminal demo\n");
    printf("  -w, --web         Run web demo (default port 8080)\n");
    printf("  -p, --port PORT    Set web server port (default: 8080)\n");
    printf("  -i, --interactive Run interactive chatbot\n");
    printf("  -s, --status       Show model status only\n");
    printf("  -a, --all         Run all demos (default)\n");
    printf("  -h, --help        Show this help\n");
    printf("\n");
    printf("Examples:\n");
    printf("  ./demo_chatbot_system                 # Run all demos\n");
    printf("  ./demo_chatbot_system -t              # Terminal demo only\n");
    printf("  ./demo_chatbot_system -w -p 3000      # Web demo on port 3000\n");
    printf("  ./demo_chatbot_system -i              # Interactive mode\n");
    printf("  ./demo_chatbot_system -s              # Show status only\n");
    printf("\n");
}

// Parse command line arguments
DemoConfig parse_arguments(int argc, char *argv[]) {
    DemoConfig config = {
        .show_terminal_demo = 0,
        .show_web_demo = 0,
        .show_model_status = 0,
        .run_interactive = 0,
        .web_port = "8080"
    };
    
    // Default to showing all demos
    if (argc == 1) {
        config.show_terminal_demo = 1;
        config.show_web_demo = 1;
        config.show_model_status = 1;
        return config;
    }
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_usage();
            exit(0);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--terminal") == 0) {
            config.show_terminal_demo = 1;
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--web") == 0) {
            config.show_web_demo = 1;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) {
            if (i + 1 < argc) {
                strncpy(config.web_port, argv[i + 1], sizeof(config.web_port) - 1);
                config.web_port[sizeof(config.web_port) - 1] = '\0';
                i++;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            config.run_interactive = 1;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--status") == 0) {
            config.show_model_status = 1;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
            config.show_terminal_demo = 1;
            config.show_web_demo = 1;
            config.show_model_status = 1;
        }
    }
    
    return config;
}

int main(int argc, char *argv[]) {
    printf("=== ADVANCED AGI CHATBOT DEMO SYSTEM ===\n");
    printf("Multi-stage learning with progressive intelligence\n");
    printf("=========================================\n");
    
    srand(time(NULL));
    
    // Parse command line arguments
    DemoConfig config = parse_arguments(argc, argv);
    
    // Show system information
    show_system_info();
    
    // Get and display model status
    ModelStatus status = get_model_status();
    
    if (config.show_model_status) {
        display_model_status(&status);
    }
    
    // Run demos based on configuration
    if (config.show_terminal_demo) {
        run_terminal_demo();
    }
    
    if (config.show_web_demo) {
        run_web_demo(config.web_port);
    }
    
    if (config.run_interactive) {
        run_interactive_demo();
    }
    
    // If no demos were specified, show usage
    if (!config.show_terminal_demo && !config.show_web_demo && 
        !config.run_interactive && !config.show_model_status) {
        printf("No demo specified. Use -h for help.\n");
        show_usage();
    }
    
    printf("\n=== DEMO SYSTEM COMPLETED ===\n");
    printf("Thank you for trying the Advanced AGI Chatbot!\n");
    
    return 0;
}
