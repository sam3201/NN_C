/*
 * Pure C Multi-Agent Orchestrator Header
 * Message passing and knowledge distillation system
 */

#ifndef MULTI_AGENT_ORCHESTRATOR_C_H
#define MULTI_AGENT_ORCHESTRATOR_C_H

#include <stddef.h>
#include <pthread.h>
#include <Python.h> // Required for PyObject
#include "specialized_agents_c.h" // Include specialized agents header for agent structs
extern AgentRegistry *global_agents;


// Forward declarations (these are now defined in specialized_agents_c.h)
// typedef struct NEAT_t NEAT_t;
// typedef struct Transformer_t Transformer_t;

// SAM core API (from libsam_core) - still needed
typedef struct SAM_t {
    int input_dim;
    int hidden_dim;
    int layers;
    long double *weights;
    long double lr;
} SAM_t;
SAM_t *SAM_init(int input_dim, int hidden_dim, int layers, int flags);
long double *SAM_forward(SAM_t *sam, long double *input, int steps);
void SAM_train(SAM_t *sam, long double *input, int steps, long double *target);
void SAM_destroy(SAM_t *sam);

// ================================
// MESSAGE SYSTEM STRUCTURES
// ================================

typedef enum {
    MSG_TASK_ASSIGNMENT,
    MSG_KNOWLEDGE_DISTILLATION,
    MSG_STATUS_UPDATE,
    MSG_RESOURCE_REQUEST
} MessageType;

typedef struct {
    MessageType type;
    char *sender;
    char *recipient;
    void *payload;
    size_t payload_size;
    time_t timestamp;
} SubmodelMessage;

// ================================
// AGENT INTERFACES (GENERIC WRAPPER)
// ================================

// Enum to identify the type of agent being orchestrated
typedef enum {
    AGENT_TYPE_UNKNOWN,
    AGENT_TYPE_RESEARCHER,
    AGENT_TYPE_CODE_WRITER,
    AGENT_TYPE_FINANCIAL,
    AGENT_TYPE_SURVIVAL,
    AGENT_TYPE_META
} AgentType;

typedef struct SubmodelAgentStruct {
    char *name;
    AgentType type; // New field to identify agent type
    void *agent_instance; // Pointer to the actual agent struct (e.g., ResearcherAgent*)
    char *capabilities[10];
    int capability_count;
    double performance_score;
    int is_active;
} SubmodelAgentStruct;

typedef SubmodelAgentStruct SubmodelAgent_t;

// ================================
// KNOWLEDGE DISTILLATION
// ================================

typedef struct {
    char *task_type;
    double success_rate;
    void *learned_patterns;
    void *performance_metrics;
    time_t distillation_time;
} DistilledKnowledge;

typedef struct {
    DistilledKnowledge **knowledge_items;
    size_t item_count;
    size_t capacity;
    SAM_t *fusion_model; // Changed void* to SAM_t*
} KnowledgeBase;

// ================================
// MULTI-AGENT ORCHESTRATOR
// ================================

typedef struct {
    char *name;
    AgentRegistry *agent_registry; // Pointer to the central registry from specialized_agents_c
    SubmodelAgent_t **submodels; // Now stores wrappers around agents from agent_registry
    size_t submodel_count;

    // Message queues
    SubmodelMessage **message_queue;
    size_t queue_size;
    size_t queue_capacity;
    pthread_mutex_t queue_mutex;

    // Knowledge distillation
    KnowledgeBase *knowledge_base;

    // Orchestration logic
    SAM_t *orchestrator_brain; // Changed void* to SAM_t*
    void **evolution_models;  // NEAT_t array (still generic as NEAT_t is opaque here)

    // Performance tracking
    double *performance_history;
    size_t history_length;
} MultiAgentOrchestrator;

// ================================
// ORCHESTRATOR API
// ================================

// Create multi-agent orchestrator
MultiAgentOrchestrator *multi_agent_orchestrator_create(const char *name);

// Add agent to orchestrator (now adds a wrapper)
int multi_agent_orchestrator_add_agent(MultiAgentOrchestrator *orchestrator, SubmodelAgent_t *agent);

// Start orchestrator
int multi_agent_orchestrator_start(MultiAgentOrchestrator *orchestrator);

// Free orchestrator
void multi_agent_orchestrator_free(MultiAgentOrchestrator *orchestrator);

// Create multi-agent system
MultiAgentOrchestrator *create_multi_agent_system();

// Get orchestrator status (now returns dict of agent statuses)
PyObject *get_orchestrator_status(MultiAgentOrchestrator *orchestrator);

#endif // MULTI_AGENT_ORCHESTRATOR_C_H
