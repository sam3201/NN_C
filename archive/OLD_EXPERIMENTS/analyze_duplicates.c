#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_PATH 1024
#define MAX_FILES 100

// File info structure
typedef struct {
    char path[MAX_PATH];
    char name[256];
    size_t size;
    time_t modified;
} FileInfo;

// Duplicate analysis
typedef struct {
    FileInfo files[MAX_FILES];
    int file_count;
    int duplicates_found;
} DuplicateAnalysis;

// Check if file is executable
int is_executable(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0 && (st.st_mode & 0111) != 0);
}

// Get file size
size_t get_file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return st.st_size;
    }
    return 0;
}

// Get file modification time
time_t get_file_modified(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return st.st_mtime;
    }
    return 0;
}

// Scan directory for files
void scan_directory(const char *path, FileInfo *files, int *count, const char *pattern) {
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir(path);
    if (dir == NULL) return;
    
    while ((entry = readdir(dir)) != NULL && *count < MAX_FILES) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char full_path[MAX_PATH];
        snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
        
        struct stat st;
        if (stat(full_path, &st) == 0 && S_ISREG(st.st_mode)) {
            // Check if filename contains pattern
            if (pattern == NULL || strstr(entry->d_name, pattern) != NULL) {
                strcpy(files[*count].path, full_path);
                strcpy(files[*count].name, entry->d_name);
                files[*count].size = get_file_size(full_path);
                files[*count].modified = get_file_modified(full_path);
                (*count)++;
            }
        }
    }
    
    closedir(dir);
}

// Find potential duplicates
void find_duplicates(FileInfo *files, int count) {
    printf("\n🔍 Looking for potential duplicates...\n");
    
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            // Check for similar names
            if (strstr(files[i].name, files[j].name) != NULL || 
                strstr(files[j].name, files[i].name) != NULL) {
                
                printf("📄 Potential duplicate:\n");
                printf("   %s (%zu bytes)\n", files[i].path, files[i].size);
                printf("   %s (%zu bytes)\n", files[j].path, files[j].size);
                printf("\n");
            }
        }
    }
}

// Analyze chatbot files specifically
void analyze_chatbot_files() {
    printf("🤖 Analyzing Chatbot Files...\n");
    
    FileInfo chatbot_files[MAX_FILES];
    int chatbot_count = 0;
    
    // Scan main chatbot files
    scan_directory("ORGANIZED/CHATBOT", chatbot_files, &chatbot_count, NULL);
    
    // Scan LLM project
    scan_directory("ORGANIZED/PROJECTS/LLM", chatbot_files, &chatbot_count, NULL);
    
    // Scan root directory for any remaining chatbot files
    scan_directory(".", chatbot_files, &chatbot_count, "chatbot");
    
    printf("Found %d chatbot-related files:\n", chatbot_count);
    
    for (int i = 0; i < chatbot_count; i++) {
        printf("  📄 %s (%zu bytes, %s)\n", 
               chatbot_files[i].name, 
               chatbot_files[i].size,
               is_executable(chatbot_files[i].path) ? "executable" : "source");
    }
    
    find_duplicates(chatbot_files, chatbot_count);
}

// Analyze model files
void analyze_model_files() {
    printf("\n📝 Analyzing Model Files...\n");
    
    FileInfo model_files[MAX_FILES];
    int model_count = 0;
    
    // Scan all model directories
    scan_directory("ORGANIZED/MODELS/STAGE1", model_files, &model_count, NULL);
    scan_directory("ORGANIZED/MODELS/STAGE2", model_files, &model_count, NULL);
    scan_directory("ORGANIZED/MODELS/STAGE3", model_files, &model_count, NULL);
    scan_directory("ORGANIZED/MODELS/STAGE4", model_files, &model_count, NULL);
    scan_directory("ORGANIZED/MODELS/STAGE5", model_files, &model_count, NULL);
    
    // Scan root for any remaining model files
    scan_directory(".", model_files, &model_count, "stage");
    scan_directory(".", model_files, &model_count, "sam_");
    
    printf("Found %d model-related files:\n", model_count);
    
    for (int i = 0; i < model_count; i++) {
        printf("  📄 %s (%zu bytes)\n", model_files[i].name, model_files[i].size);
    }
    
    find_duplicates(model_files, model_count);
}

// Analyze executable files
void analyze_executable_files() {
    printf("\n🚀 Analyzing Executable Files...\n");
    
    FileInfo exec_files[MAX_FILES];
    int exec_count = 0;
    
    // Scan all directories for executables
    scan_directory("ORGANIZED/CHATBOT", exec_files, &exec_count, NULL);
    scan_directory("ORGANIZED/TRAINING", exec_files, &exec_count, NULL);
    scan_directory("ORGANIZED/TESTS", exec_files, &exec_count, NULL);
    scan_directory("ORGANIZED/PROJECTS/LLM", exec_files, &exec_count, NULL);
    scan_directory(".", exec_files, &exec_count, NULL);
    
    printf("Found %d executable files:\n", exec_count);
    
    for (int i = 0; i < exec_count; i++) {
        if (is_executable(exec_files[i].path)) {
            printf("  🚀 %s (%zu bytes)\n", exec_files[i].name, exec_files[i].size);
        }
    }
}

// Analyze documentation files
void analyze_documentation_files() {
    printf("\n📚 Analyzing Documentation Files...\n");
    
    FileInfo doc_files[MAX_FILES];
    int doc_count = 0;
    
    // Scan all directories for documentation
    scan_directory("ORGANIZED/DOCS", doc_files, &doc_count, NULL);
    scan_directory(".", doc_files, &doc_count, ".md");
    
    printf("Found %d documentation files:\n", doc_count);
    
    for (int i = 0; i < doc_count; i++) {
        printf("  📚 %s (%zu bytes)\n", doc_files[i].name, doc_files[i].size);
    }
    
    // Check for duplicate documentation
    find_duplicates(doc_files, doc_count);
}

// Generate consolidation recommendations
void generate_recommendations() {
    printf("\n💡 Consolidation Recommendations:\n");
    printf("=====================================\n");
    
    printf("\n🤖 Chatbot Files:\n");
    printf("  • Keep: ORGANIZED/CHATBOT/TERMINAL/full_llm_chatbot (main system)\n");
    printf("  • Keep: ORGANIZED/CHATBOT/WEB/web_server (web interface)\n");
    printf("  • Keep: ORGANIZED/CHATBOT/DEMO/demo_chatbot_system (demo system)\n");
    printf("  • Consider: ORGANIZED/PROJECTS/LLM/chatbot.c (raylib GUI version)\n");
    printf("    → Move to: ORGANIZED/CHATBOT/GUI/ (if different functionality)\n");
    printf("    → Remove: If redundant with main chatbot\n");
    
    printf("\n📝 Model Files:\n");
    printf("  • Keep: All files in ORGANIZED/MODELS/ (organized by stage)\n");
    printf("  • Remove: Any duplicate model files in root directory\n");
    printf("  • Consolidate: Multiple versions of same model (keep latest)\n");
    
    printf("\n📚 Documentation Files:\n");
    printf("  • Keep: ORGANIZED/DOCS/ (main documentation)\n");
    printf("  • Remove: Duplicate README files in subdirectories\n");
    printf("  • Consolidate: Similar reports into single comprehensive docs\n");
    
    printf("\n🚀 Executable Files:\n");
    printf("  • Keep: Main executables in their respective directories\n");
    printf("  • Remove: Old/unused executables in root directory\n");
    printf("  • Consolidate: Multiple versions of same executable\n");
    
    printf("\n🔧 Utility Files:\n");
    printf("  • Keep: ORGANIZED/UTILS/ (main utilities)\n");
    printf("  • Remove: Duplicate utility files in other directories\n");
    printf("  • Consolidate: Similar scripts and tools\n");
}

int main() {
    printf("=== NN_C DUPLICATE ANALYSIS AND CONSOLIDATION ===\n");
    printf("Analyzing files for duplicates and unnecessary items...\n\n");
    
    // Analyze different file types
    analyze_chatbot_files();
    analyze_model_files();
    analyze_executable_files();
    analyze_documentation_files();
    
    // Generate recommendations
    generate_recommendations();
    
    printf("\n=== ANALYSIS COMPLETE ===\n");
    printf("Review the recommendations above and consolidate as needed.\n");
    
    return 0;
}
