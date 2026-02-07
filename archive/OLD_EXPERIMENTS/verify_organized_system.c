#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_PATH 1024

// Directory structure verification
typedef struct {
    char path[MAX_PATH];
    int exists;
    int file_count;
    int dir_count;
} DirectoryInfo;

// System status
typedef struct {
    int chatbot_terminal;
    int chatbot_web;
    int chatbot_demo;
    int models_stage1;
    int models_stage2;
    int models_stage3;
    int models_stage4;
    int models_stage5;
    int training;
    int utils;
    int docs;
    int tests;
    int projects;
    int total_files;
    int total_dirs;
} SystemStatus;

// Check if directory exists
int directory_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
}

// Check if file exists
int file_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISREG(st.st_mode));
}

// Count files and directories in directory
void count_directory_contents(const char *path, int *file_count, int *dir_count) {
    DIR *dir;
    struct dirent *entry;
    
    *file_count = 0;
    *dir_count = 0;
    
    dir = opendir(path);
    if (dir == NULL) return;
    
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char full_path[MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                (*dir_count)++;
            } else {
                (*file_count)++;
            }
        }
    }
    
    closedir(dir);
}

// Verify chatbot components
void verify_chatbot(SystemStatus *status) {
    printf("🤖 Verifying Chatbot Components...\n");
    
    // Terminal chatbot
    if (directory_exists("ORGANIZED/CHATBOT/TERMINAL")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/CHATBOT/TERMINAL", &files, &dirs);
        status->chatbot_terminal = (files >= 2); // executable + source
        printf("  ✅ Terminal Chatbot: %d files\n", files);
    } else {
        printf("  ❌ Terminal Chatbot: Directory not found\n");
    }
    
    // Web chatbot
    if (directory_exists("ORGANIZED/CHATBOT/WEB")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/CHATBOT/WEB", &files, &dirs);
        status->chatbot_web = (files >= 3); // server + source + html
        printf("  ✅ Web Chatbot: %d files\n", files);
    } else {
        printf("  ❌ Web Chatbot: Directory not found\n");
    }
    
    // Demo system
    if (directory_exists("ORGANIZED/CHATBOT/DEMO")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/CHATBOT/DEMO", &files, &dirs);
        status->chatbot_demo = (files >= 2); // executable + source
        printf("  ✅ Demo System: %d files\n", files);
    } else {
        printf("  ❌ Demo System: Directory not found\n");
    }
}

// Verify model components
void verify_models(SystemStatus *status) {
    printf("📝 Verifying Model Components...\n");
    
    // Stage 1
    if (directory_exists("ORGANIZED/MODELS/STAGE1")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/MODELS/STAGE1", &files, &dirs);
        status->models_stage1 = (files >= 10); // models + training
        printf("  ✅ Stage 1 Models: %d files\n", files);
    } else {
        printf("  ❌ Stage 1 Models: Directory not found\n");
    }
    
    // Stage 2
    if (directory_exists("ORGANIZED/MODELS/STAGE2")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/MODELS/STAGE2", &files, &dirs);
        status->models_stage2 = (files >= 5); // models + vocabulary
        printf("  ✅ Stage 2 Models: %d files\n", files);
    } else {
        printf("  ❌ Stage 2 Models: Directory not found\n");
    }
    
    // Stage 3
    if (directory_exists("ORGANIZED/MODELS/STAGE3")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/MODELS/STAGE3", &files, &dirs);
        status->models_stage3 = (files >= 5); // models + phrases
        printf("  ✅ Stage 3 Models: %d files\n", files);
    } else {
        printf("  ❌ Stage 3 Models: Directory not found\n");
    }
    
    // Stage 4
    if (directory_exists("ORGANIZED/MODELS/STAGE4")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/MODELS/STAGE4", &files, &dirs);
        status->models_stage4 = (files >= 5); // models + training
        printf("  ✅ Stage 4 Models: %d files\n", files);
    } else {
        printf("  ❌ Stage 4 Models: Directory not found\n");
    }
    
    // Stage 5
    if (directory_exists("ORGANIZED/MODELS/STAGE5")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/MODELS/STAGE5", &files, &dirs);
        status->models_stage5 = (files >= 3); // AGI components
        printf("  ✅ Stage 5 Models: %d files\n", files);
    } else {
        printf("  ❌ Stage 5 Models: Directory not found\n");
    }
}

// Verify other components
void verify_other_components(SystemStatus *status) {
    printf("🔧 Verifying Other Components...\n");
    
    // Training
    if (directory_exists("ORGANIZED/TRAINING")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/TRAINING", &files, &dirs);
        status->training = (files >= 3);
        printf("  ✅ Training: %d files\n", files);
    } else {
        printf("  ❌ Training: Directory not found\n");
    }
    
    // Utils
    if (directory_exists("ORGANIZED/UTILS")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/UTILS", &files, &dirs);
        status->utils = (files >= 10);
        printf("  ✅ Utils: %d files\n", files);
    } else {
        printf("  ❌ Utils: Directory not found\n");
    }
    
    // Docs
    if (directory_exists("ORGANIZED/DOCS")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/DOCS", &files, &dirs);
        status->docs = (files >= 5);
        printf("  ✅ Docs: %d files\n", files);
    } else {
        printf("  ❌ Docs: Directory not found\n");
    }
    
    // Tests
    if (directory_exists("ORGANIZED/TESTS")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/TESTS", &files, &dirs);
        status->tests = (files >= 5);
        printf("  ✅ Tests: %d files\n", files);
    } else {
        printf("  ❌ Tests: Directory not found\n");
    }
    
    // Projects
    if (directory_exists("ORGANIZED/PROJECTS")) {
        int files, dirs;
        count_directory_contents("ORGANIZED/PROJECTS", &files, &dirs);
        status->projects = (dirs >= 5); // project directories
        printf("  ✅ Projects: %d directories\n", dirs);
    } else {
        printf("  ❌ Projects: Directory not found\n");
    }
}

// Count total files and directories
void count_total_contents(SystemStatus *status) {
    printf("📊 Counting Total Contents...\n");
    
    status->total_files = 0;
    status->total_dirs = 0;
    
    // Count in ORGANIZED directory
    if (directory_exists("ORGANIZED")) {
        int files, dirs;
        count_directory_contents("ORGANIZED", &files, &dirs);
        status->total_files += files;
        status->total_dirs += dirs;
        
        // Recursively count subdirectories
        DIR *dir = opendir("ORGANIZED");
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                    char sub_path[MAX_PATH];
                    snprintf(sub_path, sizeof(sub_path), "ORGANIZED/%s", entry->d_name);
                    
                    DIR *sub_dir = opendir(sub_path);
                    if (sub_dir) {
                        struct dirent *sub_entry;
                        while ((sub_entry = readdir(sub_dir)) != NULL) {
                            if (sub_entry->d_type == DT_DIR && strcmp(sub_entry->d_name, ".") != 0 && strcmp(sub_entry->d_name, "..") != 0) {
                                status->total_dirs++;
                            } else if (sub_entry->d_type == DT_REG) {
                                status->total_files++;
                            }
                        }
                        closedir(sub_dir);
                    }
                }
            }
            closedir(dir);
        }
    }
    
    printf("  📁 Total Directories: %d\n", status->total_dirs);
    printf("  📄 Total Files: %d\n", status->total_files);
}

// Display system status
void display_system_status(SystemStatus *status) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    SYSTEM VERIFICATION REPORT                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    printf("\n🤖 Chatbot Components:\n");
    printf("  Terminal: %s\n", status->chatbot_terminal ? "✅ READY" : "❌ MISSING");
    printf("  Web:      %s\n", status->chatbot_web ? "✅ READY" : "❌ MISSING");
    printf("  Demo:     %s\n", status->chatbot_demo ? "✅ READY" : "❌ MISSING");
    
    printf("\n📝 Model Components:\n");
    printf("  Stage 1: %s\n", status->models_stage1 ? "✅ READY" : "❌ MISSING");
    printf("  Stage 2: %s\n", status->models_stage2 ? "✅ READY" : "❌ MISSING");
    printf("  Stage 3: %s\n", status->models_stage3 ? "✅ READY" : "❌ MISSING");
    printf("  Stage 4: %s\n", status->models_stage4 ? "✅ READY" : "❌ MISSING");
    printf("  Stage 5: %s\n", status->models_stage5 ? "✅ READY" : "❌ MISSING");
    
    printf("\n🔧 Other Components:\n");
    printf("  Training: %s\n", status->training ? "✅ READY" : "❌ MISSING");
    printf("  Utils:    %s\n", status->utils ? "✅ READY" : "❌ MISSING");
    printf("  Docs:     %s\n", status->docs ? "✅ READY" : "❌ MISSING");
    printf("  Tests:    %s\n", status->tests ? "✅ READY" : "❌ MISSING");
    printf("  Projects: %s\n", status->projects ? "✅ READY" : "❌ MISSING");
    
    printf("\n📊 Overall Statistics:\n");
    printf("  Total Directories: %d\n", status->total_dirs);
    printf("  Total Files: %d\n", status->total_files);
    
    // Calculate overall status
    int total_components = 13;
    int ready_components = status->chatbot_terminal + status->chatbot_web + status->chatbot_demo +
                          status->models_stage1 + status->models_stage2 + status->models_stage3 +
                          status->models_stage4 + status->models_stage5 + status->training +
                          status->utils + status->docs + status->tests + status->projects;
    
    double completion_rate = (double)ready_components / total_components * 100.0;
    
    printf("\n🎯 Overall Status:\n");
    printf("  Components Ready: %d/%d (%.1f%%)\n", ready_components, total_components, completion_rate);
    
    if (completion_rate >= 90.0) {
        printf("  🎉 EXCELLENT: System is fully organized and ready!\n");
    } else if (completion_rate >= 75.0) {
        printf("  ✅ GOOD: System is mostly organized.\n");
    } else if (completion_rate >= 50.0) {
        printf("  ⚠️  FAIR: System needs more organization.\n");
    } else {
        printf("  ❌ POOR: System requires significant organization.\n");
    }
    
    printf("\n🚀 Quick Start Commands:\n");
    if (status->chatbot_terminal) {
        printf("  Terminal Chatbot: cd ORGANIZED/CHATBOT/TERMINAL && ./full_llm_chatbot\n");
    }
    if (status->chatbot_web) {
        printf("  Web Chatbot:     cd ORGANIZED/CHATBOT/WEB && ./web_server\n");
    }
    if (status->chatbot_demo) {
        printf("  Demo System:     cd ORGANIZED/CHATBOT/DEMO && ./demo_chatbot_system\n");
    }
}

int main() {
    printf("=== NN_C ORGANIZED SYSTEM VERIFICATION ===\n");
    printf("Checking organized directory structure and components...\n\n");
    
    SystemStatus status = {0};
    
    // Verify all components
    verify_chatbot(&status);
    verify_models(&status);
    verify_other_components(&status);
    count_total_contents(&status);
    
    // Display final status
    display_system_status(&status);
    
    printf("\n=== VERIFICATION COMPLETE ===\n");
    
    return 0;
}
