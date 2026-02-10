/*
 * Pure C Specialized Agents Header
 * Research, Code Writing, Financial, Survival, and Meta agents
 */

#ifndef SPECIALIZED_AGENTS_C_H
#define SPECIALIZED_AGENTS_C_H

#include <stddef.h>
#include <Python.h> // Required for PyObject declarations

typedef struct NEAT_t NEAT_t;
typedef struct Transformer_t Transformer_t;

// ================================
// AGENT BASE STRUCTURE
// ================================

typedef struct {
    char *name;
    char *capabilities[10];
    int capability_count;
    double performance_score;
    int is_active;
} AgentBase;

// ================================
// INDIVIDUAL AGENT STRUCTURES
// ================================

typedef struct {
    AgentBase base;
    char **search_history;
    size_t history_count;
    size_t history_capacity;
    double credibility_score;
    char *current_search_query;
    char **found_sources;
    size_t source_count;
} ResearcherAgent;

typedef struct {
    AgentBase base;
    Transformer_t *code_transformer;
    char **generated_code;
    size_t code_count;
    size_t code_capacity;
    double code_quality_score;
    char *current_task;
    char **code_patterns;
    size_t pattern_count;
} CodeWriterAgent;

typedef struct {
    AgentBase base;
    NEAT_t *market_model;
    double *portfolio_performance;
    size_t performance_count;
    double current_portfolio_value;
    char **trading_history;
    size_t trade_count;
} FinancialAgent;

typedef struct {
    AgentBase base;
    double *threat_assessment;
    size_t threat_count;
    double survival_score;
    char **contingency_plans;
    size_t plan_count;
} SurvivalAgent;

typedef struct {
    AgentBase base;
    Transformer_t *analysis_transformer;
    char **code_improvements;
    size_t improvement_count;
    size_t improvement_capacity;
    char *current_analysis_target;
    double system_health_score;
    char **identified_issues;
    size_t issue_count;
} MetaAgent;

// ================================
// AGENT REGISTRY
// ================================

typedef struct {
    ResearcherAgent *researcher;
    CodeWriterAgent *coder;
    FinancialAgent *financer;
    SurvivalAgent *survivor;
    MetaAgent *meta;
} AgentRegistry;

// Global Agent Registry (extern to be accessed by orchestrator)
extern AgentRegistry *global_agents __attribute__((visibility("default")));

// ================================
// PREBUILT MODEL STRUCTURES (Moved from .c to .h)
// ================================

// Coherency model for maintaining conversation coherence
typedef struct {
    char *model_name;
    double coherence_threshold;
    double *coherence_history;
    size_t history_length;
    // Prebuilt model weights/parameters would go here
    double *attention_weights;
    double *memory_embeddings;
} CoherencyModel;

// Teacher model for learning and knowledge transfer
typedef struct {
    char *model_name;
    double learning_rate;
    double *knowledge_base;
    size_t knowledge_size;
    // Prebuilt model parameters
    double *teaching_weights;
    double *student_adaptation;
} TeacherModel;

// Bug-fixing model for identifying and repairing code issues
typedef struct {
    char *model_name;
    double confidence_threshold;
    double *bug_patterns;
    size_t pattern_count;
    // Prebuilt model parameters for bug detection and fixing
    double *detection_weights;
    double *repair_weights;
    char **known_bug_types;
} BugFixingModel;


// ================================
// AGENT API FUNCTIONS
// ================================

// Create individual agents
ResearcherAgent *research_agent_create() __attribute__((visibility("default")));
CodeWriterAgent *code_writer_agent_create() __attribute__((visibility("default")));
FinancialAgent *financial_agent_create() __attribute__((visibility("default")));
SurvivalAgent *survival_agent_create() __attribute__((visibility("default")));
MetaAgent *meta_agent_create() __attribute__((visibility("default")));

// Free individual agents
void research_agent_free(ResearcherAgent *agent) __attribute__((visibility("default")));
void code_writer_agent_free(CodeWriterAgent *agent) __attribute__((visibility("default")));
void financial_agent_free(FinancialAgent *agent) __attribute__((visibility("default")));
void survival_agent_free(SurvivalAgent *agent) __attribute__((visibility("default")));
void meta_agent_free(MetaAgent *agent) __attribute__((visibility("default")));

// Agent execution functions
char *research_agent_perform_search(ResearcherAgent *agent, const char *query) __attribute__((visibility("default")));
char *code_writer_agent_generate_code(CodeWriterAgent *agent, const char *spec) __attribute__((visibility("default")));
char *financial_agent_analyze_market(FinancialAgent *agent, const char *market_data) __attribute__((visibility("default")));
char *survival_agent_assess_threats(SurvivalAgent *agent) __attribute__((visibility("default")));
char *meta_agent_analyze_system(MetaAgent *agent, const char *component) __attribute__((visibility("default")));

// Data analysis functions
char *research_agent_analyze_data(ResearcherAgent *agent, const char *data) __attribute__((visibility("default")));
char *code_writer_agent_analyze_code(CodeWriterAgent *agent, const char *code) __attribute__((visibility("default")));

// Agent registry functions
AgentRegistry *agent_registry_create() __attribute__((visibility("default")));
void agent_registry_free(AgentRegistry *registry) __attribute__((visibility("default")));

// Coherency/Teacher Model API functions
CoherencyModel *coherency_model_create() __attribute__((visibility("default")));
TeacherModel *teacher_model_create() __attribute__((visibility("default")));
double coherency_model_evaluate(const char *conversation_history, const char *new_message) __attribute__((visibility("default")));
char *teacher_model_generate_lesson(const char *topic, const char *student_level) __attribute__((visibility("default")));
void coherency_model_free(CoherencyModel *model) __attribute__((visibility("default")));
void teacher_model_free(TeacherModel *model) __attribute__((visibility("default")));

// Bug-Fixing Model API functions
BugFixingModel *bug_fixing_model_create() __attribute__((visibility("default")));
char *bug_fixing_model_analyze_code(const char *code_snippet, const char *error_message) __attribute__((visibility("default")));
char *bug_fixing_model_generate_fix(const char *code_snippet, const char *bug_description) __attribute__((visibility("default")));
void bug_fixing_model_free(BugFixingModel *model) __attribute__((visibility("default")));

// Declare global instances of prebuilt models
extern CoherencyModel *global_coherency_model __attribute__((visibility("default")));
extern TeacherModel *global_teacher_model __attribute__((visibility("default")));
extern BugFixingModel *global_bug_fixing_model __attribute__((visibility("default")));

// Declare Python web search integration components
extern PyObject *pSamWebSearchModule __attribute__((visibility("default")));
extern PyObject *pSearchWebWithSamFunc __attribute__((visibility("default")));
extern int sam_web_search_is_initialized __attribute__((visibility("default")));
extern int init_python_web_search() __attribute__((visibility("default")));

// Safety helpers (static function in .c, but prototype needed for consistency if it were public)
// For now, keep it internal to .c since it's only used there.
// extern void safe_copy(char *dest, size_t dest_size, const char *src);

#endif // SPECIALIZED_AGENTS_C_H