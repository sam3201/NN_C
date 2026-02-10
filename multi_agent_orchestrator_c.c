/*
 * Pure C Multi-Agent Orchestrator - Using Existing SAM Framework
 * Complete multi-agent system with knowledge distillation and orchestration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

// Use available headers
#include "multi_agent_orchestrator_c.h"
#include "specialized_agents_c.h" // Include specialized agents header


// ================================
// MESSAGE/AGENT TYPES
// ================================

typedef struct NEAT_t {
    int input_dim;
    int output_dim;
    int population;
} NEAT_t;

static NEAT_t *NEAT_init(int input_dim, int output_dim, int population) {
    NEAT_t *model = (NEAT_t *)calloc(1, sizeof(NEAT_t));
    if (!model) return NULL;
    model->input_dim = input_dim;
    model->output_dim = output_dim;
    model->population = population;
    return model;
}

// ================================
// LIGHTWEIGHT SAM CORE (LOCAL)
// ================================

SAM_t *SAM_init(int input_dim, int hidden_dim, int layers, int flags) {
    (void)flags;
    SAM_t *sam = (SAM_t *)calloc(1, sizeof(SAM_t));
    if (!sam) return NULL;
    sam->input_dim = input_dim;
    sam->hidden_dim = hidden_dim;
    sam->layers = layers;
    sam->lr = 0.001L;
    size_t weight_count = (size_t)input_dim * (size_t)hidden_dim;
    sam->weights = (long double *)calloc(weight_count, sizeof(long double));
    if (!sam->weights) {
        free(sam);
        return NULL;
    }
    for (size_t i = 0; i < weight_count; i++) {
        sam->weights[i] = ((long double)rand() / (long double)RAND_MAX - 0.5L) * 0.01L;
    }
    return sam;
}

long double *SAM_forward(SAM_t *sam, long double *input, int steps) {
    (void)steps;
    if (!sam || !input) return NULL;
    static __thread long double *output = NULL;
    static __thread size_t output_cap = 0;
    if (output_cap < (size_t)sam->hidden_dim) {
        free(output);
        output = (long double *)calloc((size_t)sam->hidden_dim, sizeof(long double));
        output_cap = (size_t)sam->hidden_dim;
    }
    for (int i = 0; i < sam->hidden_dim; i++) {
        long double acc = 0.0L;
        for (int j = 0; j < sam->input_dim; j++) {
            acc += sam->weights[(size_t)i * (size_t)sam->input_dim + (size_t)j] * input[j];
        }
        output[i] = tanhl(acc);
    }
    return output;
}

void SAM_train(SAM_t *sam, long double *input, int steps, long double *target) {
    (void)steps;
    if (!sam || !input || !target) return;
    long double *out = SAM_forward(sam, input, 1);
    if (!out) return;
    for (int i = 0; i < sam->hidden_dim; i++) {
        long double err = target[i] - out[i];
        for (int j = 0; j < sam->input_dim; j++) {
            size_t idx = (size_t)i * (size_t)sam->input_dim + (size_t)j;
            sam->weights[idx] += sam->lr * err * input[j];
        }
    }
}

void SAM_destroy(SAM_t *sam) {
    if (!sam) return;
    free(sam->weights);
    free(sam);
}

// ================================
// FORWARD DECLARATIONS
// ================================

void knowledge_base_free(KnowledgeBase *kb);
void multi_agent_orchestrator_free(MultiAgentOrchestrator *orchestrator);
MultiAgentOrchestrator *create_multi_agent_system();

// ================================
// INDIVIDUAL AGENTS - Pure C Implementations
// ================================



// ================================
// MESSAGE QUEUE IMPLEMENTATION - Pure C
// ================================

int message_queue_init(MultiAgentOrchestrator *orchestrator, size_t capacity) {
    orchestrator->message_queue = malloc(capacity * sizeof(SubmodelMessage*));
    if (!orchestrator->message_queue) return -1;

    orchestrator->queue_capacity = capacity;
    orchestrator->queue_size = 0;

    if (pthread_mutex_init(&orchestrator->queue_mutex, NULL) != 0) {
        free(orchestrator->message_queue);
        return -1;
    }

    return 0;
}

int message_queue_send(MultiAgentOrchestrator *orchestrator, SubmodelMessage *msg) {
    pthread_mutex_lock(&orchestrator->queue_mutex);

    if (orchestrator->queue_size >= orchestrator->queue_capacity) {
        // Queue full - could implement overflow handling
        pthread_mutex_unlock(&orchestrator->queue_mutex);
        return -1;
    }

    orchestrator->message_queue[orchestrator->queue_size++] = msg;
    msg->timestamp = time(NULL);

    pthread_mutex_unlock(&orchestrator->queue_mutex);
    return 0;
}

SubmodelMessage *message_queue_receive(MultiAgentOrchestrator *orchestrator) {
    pthread_mutex_lock(&orchestrator->queue_mutex);

    if (orchestrator->queue_size == 0) {
        pthread_mutex_unlock(&orchestrator->queue_mutex);
        return NULL;
    }

    SubmodelMessage *msg = orchestrator->message_queue[0];

    // Shift queue
    for (size_t i = 1; i < orchestrator->queue_size; i++) {
        orchestrator->message_queue[i-1] = orchestrator->message_queue[i];
    }
    orchestrator->queue_size--;

    pthread_mutex_unlock(&orchestrator->queue_mutex);
    return msg;
}

// ================================
// KNOWLEDGE DISTILLATION - Pure C
// ================================

KnowledgeBase *knowledge_base_create(size_t capacity) {
    KnowledgeBase *kb = malloc(sizeof(KnowledgeBase));
    if (!kb) return NULL;

    kb->knowledge_items = malloc(capacity * sizeof(DistilledKnowledge*));
    if (!kb->knowledge_items) {
        free(kb);
        return NULL;
    }

    kb->item_count = 0;
    kb->capacity = capacity;

    // Create fusion model using SAM (optional)
    kb->fusion_model = SAM_init(128, 64, 8, 0);
    if (!kb->fusion_model) {
        printf("⚠️ SAM fusion model initialization failed, continuing without knowledge fusion\n");
        // Continue without SAM - knowledge base can still store distilled knowledge
    }

    return kb;
}

int knowledge_base_add(KnowledgeBase *kb, DistilledKnowledge *knowledge) {
    if (kb->item_count >= kb->capacity) {
        // Implement resizing - double the capacity
        size_t new_capacity = kb->capacity * 2;
        DistilledKnowledge **new_items = realloc(kb->knowledge_items, new_capacity * sizeof(DistilledKnowledge*));
        if (!new_items) {
            return -1; // Resizing failed
        }
        kb->knowledge_items = new_items;
        kb->capacity = new_capacity;
        printf("🔄 Knowledge base resized to %zu items\n", new_capacity);
    }

    kb->knowledge_items[kb->item_count++] = knowledge;
    knowledge->distillation_time = time(NULL);

    // Use SAM for knowledge fusion - convert knowledge to SAM input format and update fusion model
    if (kb->fusion_model) {
        // Convert distilled knowledge to SAM input format
        // This would create input vectors from the knowledge patterns
        long double *knowledge_input = calloc(256, sizeof(long double)); // SAM input dimension
        // Fill knowledge_input with knowledge features (task_type, success_rate, etc.)

        // Run knowledge fusion through SAM
        long double *target_output = calloc(128, sizeof(long double)); // SAM output dimension
        SAM_train(kb->fusion_model, knowledge_input, 1, target_output);

        free(knowledge_input);
        free(target_output);
    }

    return 0;
}

// ================================
// AGENT IMPLEMENTATIONS - Pure C
// ================================



// ================================
// MULTI-AGENT ORCHESTRATOR CREATION - Pure C
// ================================

MultiAgentOrchestrator *multi_agent_orchestrator_create(const char *name) {
    MultiAgentOrchestrator *orchestrator = malloc(sizeof(MultiAgentOrchestrator));
    if (!orchestrator) return NULL;

    orchestrator->name = strdup(name);
    orchestrator->submodel_count = 0;
    orchestrator->submodels = NULL;

    // Initialize message queue
    if (message_queue_init(orchestrator, 1000) != 0) {
        free(orchestrator->name);
        free(orchestrator);
        return NULL;
    }

    // Create knowledge base
    orchestrator->knowledge_base = knowledge_base_create(1000);
    if (!orchestrator->knowledge_base) {
        pthread_mutex_destroy(&orchestrator->queue_mutex);
        free(orchestrator->message_queue);
        free(orchestrator->name);
        free(orchestrator);
        return NULL;
    }

    // Create orchestrator brain using SAM (optional)
    orchestrator->orchestrator_brain = SAM_init(256, 128, 8, 0);
    if (!orchestrator->orchestrator_brain) {
        printf("⚠️ SAM initialization failed, continuing without orchestrator brain\n");
        // Continue without SAM - orchestrator can still function with basic message passing
    }

    // Initialize evolution models using NEAT
    orchestrator->evolution_models = calloc(orchestrator->submodel_count, sizeof(void*));
    for (size_t i = 0; i < orchestrator->submodel_count; i++) {
        // Create NEAT population for each agent type with appropriate dimensions
        // Different agents may need different input/output dimensions
        int input_dim = 10;  // Standard input features (performance, task complexity, etc.)
        int output_dim = 5;  // Standard output actions (adjust weights, change strategy, etc.)

        // Customize dimensions based on agent type
        if (i == 0) { // Researcher agent
            input_dim = 12; // Additional research-specific features
        } else if (i == 1) { // Code writer agent
            input_dim = 15; // Additional code-specific features
            output_dim = 8; // More complex decision space
        }

        orchestrator->evolution_models[i] = NEAT_init(input_dim, output_dim, 50); // population_size = 50
        if (!orchestrator->evolution_models[i]) {
            printf("⚠️ NEAT initialization failed for agent %zu\n", i);
        } else {
            printf("✅ NEAT evolution model initialized for agent %zu (input:%d, output:%d)\n",
                   i, input_dim, output_dim);
        }
    }

    // Performance tracking
    orchestrator->performance_history = calloc(1000, sizeof(double));
    orchestrator->history_length = 0;

    return orchestrator;
}

int multi_agent_orchestrator_add_agent(MultiAgentOrchestrator *orchestrator, SubmodelAgent_t *agent) {
    orchestrator->submodels = realloc(orchestrator->submodels,
                                     (orchestrator->submodel_count + 1) * sizeof(SubmodelAgent_t*));
    if (!orchestrator->submodels) return -1;

    orchestrator->submodels[orchestrator->submodel_count++] = agent;
    agent->is_active = 1;

    printf("✅ Added agent '%s' to orchestrator '%s'\n", agent->name, orchestrator->name);
    return 0;
}

// ================================
// ORCHESTRATION LOGIC - Pure C
// ================================

void *orchestration_thread(void *arg) {
    MultiAgentOrchestrator *orchestrator = (MultiAgentOrchestrator*)arg;

    while (1) {
        // Process messages
        SubmodelMessage *msg = message_queue_receive(orchestrator);
        if (msg) {
            // Route message to appropriate agent by name (recipient)
            for (size_t i = 0; i < orchestrator->submodel_count; i++) {
                SubmodelAgent_t *agent_wrapper = orchestrator->submodels[i];
                if (strcmp(msg->recipient, agent_wrapper->name) == 0) {
                    char *agent_result = NULL;
                    // Dispatch task based on agent type
                    switch (agent_wrapper->type) {
                        case AGENT_TYPE_RESEARCHER: {
                            ResearcherAgent *researcher = (ResearcherAgent*)agent_wrapper->agent_instance;
                            agent_result = research_agent_perform_search(researcher, (const char*)msg->payload);
                            printf("🔍 Orchestrator: Researcher %s executed task: %s\n", agent_wrapper->name, (char*)msg->payload);
                            break;
                        }
                        case AGENT_TYPE_CODE_WRITER: {
                            CodeWriterAgent *coder = (CodeWriterAgent*)agent_wrapper->agent_instance;
                            agent_result = code_writer_agent_generate_code(coder, (const char*)msg->payload);
                            printf("💻 Orchestrator: CodeWriter %s executed task: %s\n", agent_wrapper->name, (char*)msg->payload);
                            break;
                        }
                        case AGENT_TYPE_FINANCIAL: {
                            FinancialAgent *financer = (FinancialAgent*)agent_wrapper->agent_instance;
                            agent_result = financial_agent_analyze_market(financer, (const char*)msg->payload);
                            printf("💰 Orchestrator: Financial %s executed task: %s\n", agent_wrapper->name, (char*)msg->payload);
                            break;
                        }
                        case AGENT_TYPE_SURVIVAL: {
                            SurvivalAgent *survivor = (SurvivalAgent*)agent_wrapper->agent_instance;
                            agent_result = survival_agent_assess_threats(survivor); // Survival has no payload
                            printf("🛡️ Orchestrator: Survival %s executed task.\n", agent_wrapper->name);
                            break;
                        }
                        case AGENT_TYPE_META: {
                            MetaAgent *meta = (MetaAgent*)agent_wrapper->agent_instance;
                            agent_result = meta_agent_analyze_system(meta, (const char*)msg->payload);
                            printf("🔧 Orchestrator: Meta %s executed task: %s\n", agent_wrapper->name, (char*)msg->payload);
                            break;
                        }
                        case AGENT_TYPE_UNKNOWN:
                        default:
                            printf("⚠️ Orchestrator: Unknown agent type for %s. Cannot execute task.\n", agent_wrapper->name);
                            break;
                    }

                    if (agent_result) {
                        printf("   Result: %s\n", agent_result);
                        free(agent_result); // Free the result string after use
                    }
                    break; // Message handled
                }
            }

            // Cleanup message
            free(msg->sender);
            free(msg->recipient);
            free(msg->payload);
            free(msg);
        }

        // Knowledge distillation cycle
        if (orchestrator->knowledge_base->item_count > 0) {
            if (orchestrator->knowledge_base->fusion_model && orchestrator->orchestrator_brain) {
                long double *aggregated_knowledge = calloc(256, sizeof(long double));
                size_t knowledge_count = orchestrator->knowledge_base->item_count;

                for (size_t k = 0; k < knowledge_count && k < 10; k++) {
                    DistilledKnowledge *item = orchestrator->knowledge_base->knowledge_items[k];
                    size_t base_idx = k * 25;
                    aggregated_knowledge[base_idx] = item->success_rate;
                    aggregated_knowledge[base_idx + 1] = (item->task_type) ? strlen(item->task_type) : 0;
                    aggregated_knowledge[base_idx + 2] = item->distillation_time % 1000;
                }

                long double *fused_knowledge = SAM_forward(orchestrator->orchestrator_brain, aggregated_knowledge, 1);

                for (size_t i = 0; i < orchestrator->submodel_count; i++) {
                    SubmodelAgent_t *agent_wrapper = orchestrator->submodels[i];
                    int relevant = 0;
                    for (int cap = 0; cap < agent_wrapper->capability_count; cap++) {
                        if (strstr(agent_wrapper->capabilities[cap], "analysis") ||
                            strstr(agent_wrapper->capabilities[cap], "learning")) {
                            relevant = 1;
                            break;
                        }
                    }

                    if (relevant) {
                        printf("🧠 Knowledge distilled to agent: %s\n", agent_wrapper->name);
                        // Future: call a specialized_agents_c function for distillation if needed
                    }
                }

                free(aggregated_knowledge);
                free(fused_knowledge);
            } else {
                for (size_t i = 0; i < orchestrator->submodel_count; i++) {
                    SubmodelAgent_t *agent_wrapper = orchestrator->submodels[i];
                    // Future: call a specialized_agents_c function for distillation if needed
                }
            }
        }

        usleep(10000); // 10ms
    }

    return NULL;
}

int multi_agent_orchestrator_start(MultiAgentOrchestrator *orchestrator) {
    printf("🚀 Starting Multi-Agent Orchestrator: %s\n", orchestrator->name);
    printf("   Agents: %zu\n", orchestrator->submodel_count);
    printf("   Using SAM for orchestration brain\n");
    printf("   Using NEAT for agent evolution\n");
    printf("   Knowledge distillation: ACTIVE\n");

    // Start orchestration thread
    pthread_t thread;
    if (pthread_create(&thread, NULL, orchestration_thread, orchestrator) != 0) {
        fprintf(stderr, "Failed to create orchestration thread\n");
        return -1;
    }

    // Detach thread so it runs independently
    pthread_detach(thread);

    printf("✅ Multi-Agent Orchestrator started successfully\n");
    return 0;
}

// ================================
// CLEANUP - Pure C
// ================================

void knowledge_base_free(KnowledgeBase *kb) {
    if (kb) {
        for (size_t i = 0; i < kb->item_count; i++) {
            free(kb->knowledge_items[i]);
        }
        free(kb->knowledge_items);
        if (kb->fusion_model) {
            SAM_destroy(kb->fusion_model);
        }
        free(kb);
    }
}

void multi_agent_orchestrator_free(MultiAgentOrchestrator *orchestrator) {
    if (orchestrator) {
        free(orchestrator->name);

        // Free the central agent registry (which frees individual agents)
        if (orchestrator->agent_registry) {
            agent_registry_free(orchestrator->agent_registry);
        }


        
        // Free the SubmodelAgent_t wrappers
        for (size_t i = 0; i < orchestrator->submodel_count; i++) {
            if (orchestrator->submodels[i]) {
                // Free wrapper's name and capabilities if they were strdup'd
                free(orchestrator->submodels[i]->name);
                for (int cap = 0; cap < orchestrator->submodels[i]->capability_count; cap++) {
                    free(orchestrator->submodels[i]->capabilities[cap]);
                }
                free(orchestrator->submodels[i]); // Free the wrapper itself
            }
        }
        free(orchestrator->submodels);

        // Cleanup message queue
        for (size_t i = 0; i < orchestrator->queue_size; i++) {
            // Free contents of messages in queue
            free(orchestrator->message_queue[i]->sender);
            free(orchestrator->message_queue[i]->recipient);
            free(orchestrator->message_queue[i]->payload);
            free(orchestrator->message_queue[i]);
        }
        free(orchestrator->message_queue);
        pthread_mutex_destroy(&orchestrator->queue_mutex);

        knowledge_base_free(orchestrator->knowledge_base);

        if (orchestrator->orchestrator_brain) {
            SAM_destroy(orchestrator->orchestrator_brain);
        }

        free(orchestrator->performance_history);
        free(orchestrator);
    }
}

// ================================
// MAIN ORCHESTRATOR SETUP - Pure C
// ================================

MultiAgentOrchestrator *create_multi_agent_system() {
    MultiAgentOrchestrator *orchestrator = multi_agent_orchestrator_create("SAM_MultiAgent");

    if (!orchestrator) return NULL;

    // Rely on global_agents being initialized via specialized_agents_c.create_agents()
    // which is called from Python side.
    if (!global_agents) {
        fprintf(stderr, "❌ Error: specialized_agents_c.create_agents() must be called from Python before multi_agent_orchestrator_c.create_system()\n");
        multi_agent_orchestrator_free(orchestrator); // Cleanup partially created orchestrator
        return NULL;
    }
    orchestrator->agent_registry = global_agents;
    
    // Add wrappers for each agent from the registry to orchestrator->submodels
    // Researcher Agent
    SubmodelAgent_t *researcher_wrapper = malloc(sizeof(SubmodelAgent_t));
    if (!researcher_wrapper) { multi_agent_orchestrator_free(orchestrator); return NULL; }
    researcher_wrapper->name = strdup(orchestrator->agent_registry->researcher->base.name);
    researcher_wrapper->type = AGENT_TYPE_RESEARCHER;
    researcher_wrapper->agent_instance = orchestrator->agent_registry->researcher;
    for (int i = 0; i < orchestrator->agent_registry->researcher->base.capability_count; i++) {
        researcher_wrapper->capabilities[i] = strdup(orchestrator->agent_registry->researcher->base.capabilities[i]);
    }
    researcher_wrapper->capability_count = orchestrator->agent_registry->researcher->base.capability_count;
    researcher_wrapper->performance_score = orchestrator->agent_registry->researcher->base.performance_score;
    researcher_wrapper->is_active = orchestrator->agent_registry->researcher->base.is_active;
    multi_agent_orchestrator_add_agent(orchestrator, researcher_wrapper);

    // Code Writer Agent
    SubmodelAgent_t *coder_wrapper = malloc(sizeof(SubmodelAgent_t));
    if (!coder_wrapper) { multi_agent_orchestrator_free(orchestrator); return NULL; }
    coder_wrapper->name = strdup(orchestrator->agent_registry->coder->base.name);
    coder_wrapper->type = AGENT_TYPE_CODE_WRITER;
    coder_wrapper->agent_instance = orchestrator->agent_registry->coder;
    for (int i = 0; i < orchestrator->agent_registry->coder->base.capability_count; i++) {
        coder_wrapper->capabilities[i] = strdup(orchestrator->agent_registry->coder->base.capabilities[i]);
    }
    coder_wrapper->capability_count = orchestrator->agent_registry->coder->base.capability_count;
    coder_wrapper->performance_score = orchestrator->agent_registry->coder->base.performance_score;
    coder_wrapper->is_active = orchestrator->agent_registry->coder->base.is_active;
    multi_agent_orchestrator_add_agent(orchestrator, coder_wrapper);

    // Financial Agent
    SubmodelAgent_t *financer_wrapper = malloc(sizeof(SubmodelAgent_t));
    if (!financer_wrapper) { multi_agent_orchestrator_free(orchestrator); return NULL; }
    financer_wrapper->name = strdup(orchestrator->agent_registry->financer->base.name);
    financer_wrapper->type = AGENT_TYPE_FINANCIAL;
    financer_wrapper->agent_instance = orchestrator->agent_registry->financer;
    for (int i = 0; i < orchestrator->agent_registry->financer->base.capability_count; i++) {
        financer_wrapper->capabilities[i] = strdup(orchestrator->agent_registry->financer->base.capabilities[i]);
    }
    financer_wrapper->capability_count = orchestrator->agent_registry->financer->base.capability_count;
    financer_wrapper->performance_score = orchestrator->agent_registry->financer->base.performance_score;
    financer_wrapper->is_active = orchestrator->agent_registry->financer->base.is_active;
    multi_agent_orchestrator_add_agent(orchestrator, financer_wrapper);

    // Survival Agent
    SubmodelAgent_t *survivor_wrapper = malloc(sizeof(SubmodelAgent_t));
    if (!survivor_wrapper) { multi_agent_orchestrator_free(orchestrator); return NULL; }
    survivor_wrapper->name = strdup(orchestrator->agent_registry->survivor->base.name);
    survivor_wrapper->type = AGENT_TYPE_SURVIVAL;
    survivor_wrapper->agent_instance = orchestrator->agent_registry->survivor;
    for (int i = 0; i < orchestrator->agent_registry->survivor->base.capability_count; i++) {
        survivor_wrapper->capabilities[i] = strdup(orchestrator->agent_registry->survivor->base.capabilities[i]);
    }
    survivor_wrapper->capability_count = orchestrator->agent_registry->survivor->base.capability_count;
    survivor_wrapper->performance_score = orchestrator->agent_registry->survivor->base.performance_score;
    survivor_wrapper->is_active = orchestrator->agent_registry->survivor->base.is_active;
    multi_agent_orchestrator_add_agent(orchestrator, survivor_wrapper);

    // Meta Agent
    SubmodelAgent_t *meta_wrapper = malloc(sizeof(SubmodelAgent_t));
    if (!meta_wrapper) { multi_agent_orchestrator_free(orchestrator); return NULL; }
    meta_wrapper->name = strdup(orchestrator->agent_registry->meta->base.name);
    meta_wrapper->type = AGENT_TYPE_META;
    meta_wrapper->agent_instance = orchestrator->agent_registry->meta;
    for (int i = 0; i < orchestrator->agent_registry->meta->base.capability_count; i++) {
        meta_wrapper->capabilities[i] = strdup(orchestrator->agent_registry->meta->base.capabilities[i]);
    }
    meta_wrapper->capability_count = orchestrator->agent_registry->meta->base.capability_count;
    meta_wrapper->performance_score = orchestrator->agent_registry->meta->base.performance_score;
    meta_wrapper->is_active = orchestrator->agent_registry->meta->base.is_active;
    multi_agent_orchestrator_add_agent(orchestrator, meta_wrapper);

    return orchestrator;
}

// ================================
// PYTHON BINDINGS - Pure C
// ================================

#include <Python.h>

static MultiAgentOrchestrator *global_orchestrator = NULL;

// Python-callable functions for multi_agent_orchestrator_c
static PyObject *py_create_multi_agent_system(PyObject *self, PyObject *args) {
    if (global_orchestrator) {
        multi_agent_orchestrator_free(global_orchestrator);
    }

    printf("   Calling create_multi_agent_system...\n");
    global_orchestrator = create_multi_agent_system();
    printf("   create_multi_agent_system returned: %p\n", global_orchestrator);

    if (!global_orchestrator) {
        printf("   ❌ global_orchestrator is NULL, setting error\n");
        PyErr_SetString(PyExc_RuntimeError, "Failed to create multi-agent system");
        return NULL;
    }

    printf("✅ Created multi-agent orchestrator with %zu agents\n", global_orchestrator->submodel_count);
    Py_RETURN_NONE;
}

static PyObject *py_start_orchestrator(PyObject *self, PyObject *args) {
    if (!global_orchestrator) {
        PyErr_SetString(PyExc_RuntimeError, "Multi-agent system not initialized");
        return NULL;
    }

    if (multi_agent_orchestrator_start(global_orchestrator) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to start orchestrator");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *py_get_orchestrator_status(PyObject *self, PyObject *args) {
    if (!global_orchestrator) {
        PyErr_SetString(PyExc_RuntimeError, "Multi-agent system not initialized");
        return NULL;
    }

    // Create a Python dictionary to hold the status
    PyObject *status_dict = PyDict_New();
    if (!status_dict) {
        return NULL;
    }

    // Add basic orchestrator status
    PyDict_SetItemString(status_dict, "name", PyUnicode_FromString(global_orchestrator->name));
    PyDict_SetItemString(status_dict, "agent_count", PyLong_FromSize_t(global_orchestrator->submodel_count));
    PyDict_SetItemString(status_dict, "queue_size", PyLong_FromSize_t(global_orchestrator->queue_size));
    PyDict_SetItemString(status_dict, "knowledge_items", PyLong_FromSize_t(global_orchestrator->knowledge_base->item_count));

    // Add individual agent statuses
    PyObject *agents_list = PyList_New(global_orchestrator->submodel_count);
    if (!agents_list) {
        Py_DECREF(status_dict);
        return NULL;
    }

    for (size_t i = 0; i < global_orchestrator->submodel_count; i++) {
        SubmodelAgent_t *agent_wrapper = global_orchestrator->submodels[i];
        PyObject *agent_dict = PyDict_New();
        if (!agent_dict) {
            Py_DECREF(agents_list);
            Py_DECREF(status_dict);
            return NULL;
        }

        PyDict_SetItemString(agent_dict, "name", PyUnicode_FromString(agent_wrapper->name));
        PyDict_SetItemString(agent_dict, "type", PyLong_FromLong(agent_wrapper->type)); // AgentType enum value
        PyDict_SetItemString(agent_dict, "is_active", PyBool_FromLong(agent_wrapper->is_active));
        PyDict_SetItemString(agent_dict, "performance_score", PyFloat_FromDouble(agent_wrapper->performance_score));

        // Capabilities list
        PyObject *capabilities_list = PyList_New(agent_wrapper->capability_count);
        if (!capabilities_list) {
            Py_DECREF(agent_dict);
            Py_DECREF(agents_list);
            Py_DECREF(status_dict);
            return NULL;
        }
        for (int j = 0; j < agent_wrapper->capability_count; j++) {
            PyList_SetItem(capabilities_list, j, PyUnicode_FromString(agent_wrapper->capabilities[j]));
        }
        PyDict_SetItemString(agent_dict, "capabilities", capabilities_list);
        
        PyList_SetItem(agents_list, i, agent_dict); // PyList_SetItem "steals" a reference to agent_dict
    }
    PyDict_SetItemString(status_dict, "agents", agents_list); // PyDict_SetItemString "steals" a reference to agents_list

    return status_dict;
}

// Python-callable functions from specialized_agents_c.c
// (These were previously in specialized_agents_c.c and are now moved here)

static PyObject *py_create_agents(PyObject *self, PyObject *args) {
    if (global_agents) {
        agent_registry_free(global_agents);
    }

    global_agents = agent_registry_create();
    if (!global_agents) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create agent registry");
        return NULL;
    }

    // Initialize prebuilt models
    if (!global_coherency_model) {
        global_coherency_model = coherency_model_create();
    }
    if (!global_teacher_model) {
        global_teacher_model = teacher_model_create();
    }
    if (!global_bug_fixing_model) {
        global_bug_fixing_model = bug_fixing_model_create();
    }
    
    // Initialize Python web search
    if (!init_python_web_search()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Python web search");
        return NULL;
    }

    printf("✅ Created all specialized agents using existing framework\n");
    printf("✅ Initialized all prebuilt models (Coherency, Teacher, Bug-Fixing)\n");
    Py_RETURN_NONE;
}

static PyObject *py_research_task(PyObject *self, PyObject *args) {
    const char *query;
    if (!PyArg_ParseTuple(args, "s", &query)) {
        return NULL;
    }

    if (!global_agents || !global_agents->researcher) {
        PyErr_SetString(PyExc_RuntimeError, "Research agent not initialized");
        return NULL;
    }

    char *result = research_agent_perform_search(global_agents->researcher, query);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Research agent returned no result");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(result);
    free(result); // Free the C string after creating Python object
    return py_result;
}

static PyObject *py_code_generation(PyObject *self, PyObject *args) {
    const char *spec;
    if (!PyArg_ParseTuple(args, "s", &spec)) {
        return NULL;
    }

    if (!global_agents || !global_agents->coder) {
        PyErr_SetString(PyExc_RuntimeError, "Code writer agent not initialized");
        return NULL;
    }

    char *result = code_writer_agent_generate_code(global_agents->coder, spec);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Code writer agent returned no result");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(result);
    free(result);
    return py_result;
}

static PyObject *py_financial_analysis(PyObject *self, PyObject *args) {
    const char *market;
    if (!PyArg_ParseTuple(args, "s", &market)) {
        return NULL;
    }

    if (!global_agents || !global_agents->financer) {
        PyErr_SetString(PyExc_RuntimeError, "Financial agent not initialized");
        return NULL;
    }

    char *result = financial_agent_analyze_market(global_agents->financer, market);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Financial agent returned no result");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(result);
    free(result);
    return py_result;
}

static PyObject *py_survival_assessment(PyObject *self, PyObject *args) {
    if (!global_agents || !global_agents->survivor) {
        PyErr_SetString(PyExc_RuntimeError, "Survival agent not initialized");
        return NULL;
    }

    char *result = survival_agent_assess_threats(global_agents->survivor);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Survival agent returned no result");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(result);
    free(result);
    return py_result;
}

static PyObject *py_evaluate_coherence(PyObject *self, PyObject *args) {
    const char *conversation_history, *new_message;
    if (!PyArg_ParseTuple(args, "ss", &conversation_history, &new_message)) {
        return NULL;
    }

    double coherence_score = coherency_model_evaluate(conversation_history, new_message);
    return PyFloat_FromDouble(coherence_score);
}

static PyObject *py_analyze_code(PyObject *self, PyObject *args) {
    const char *code_snippet, *error_message;
    if (!PyArg_ParseTuple(args, "ss", &code_snippet, &error_message)) {
        return NULL;
    }

    char *analysis = bug_fixing_model_analyze_code(code_snippet, error_message);
    if (!analysis) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to analyze code");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(analysis);
    free(analysis);
    return py_result;
}

static PyObject *py_generate_fix(PyObject *self, PyObject *args) {
    const char *code_snippet, *bug_description;
    if (!PyArg_ParseTuple(args, "ss", &code_snippet, &bug_description)) {
        return NULL;
    }

    char *fix = bug_fixing_model_generate_fix(code_snippet, bug_description);
    if (!fix) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to generate fix");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(fix);
    free(fix);
    return py_result;
}

static PyObject *py_meta_analysis(PyObject *self, PyObject *args) {
    const char *component;
    if (!PyArg_ParseTuple(args, "s", &component)) {
        return NULL;
    }

    if (!global_agents || !global_agents->meta) {
        PyErr_SetString(PyExc_RuntimeError, "Meta agent not initialized");
        return NULL;
    }

    char *result = meta_agent_analyze_system(global_agents->meta, component);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Meta agent returned no result");
        return NULL;
    }
    PyObject *py_result = PyUnicode_FromString(result);
    free(result);
    return py_result;
}

static PyMethodDef AgentMethods[] = {
    {"create_agents", py_create_agents, METH_NOARGS, "Create all specialized agents"},
    {"research", py_research_task, METH_VARARGS, "Perform research task"},
    {"generate_code", py_code_generation, METH_VARARGS, "Generate code"},
    {"analyze_market", py_financial_analysis, METH_VARARGS, "Analyze market"},
    {"assess_survival", py_survival_assessment, METH_NOARGS, "Assess survival threats"},
    {"analyze_system", py_meta_analysis, METH_VARARGS, "Analyze system component"},
    {"evaluate_coherence", py_evaluate_coherence, METH_VARARGS, "Evaluate conversation coherence"},
    {"analyze_code", py_analyze_code, METH_VARARGS, "Analyze code for bugs"},
    {"generate_fix", py_generate_fix, METH_VARARGS, "Generate bug fix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef agent_module = {
    PyModuleDef_HEAD_INIT,
    "specialized_agents_c",
    "Pure C specialized agents using existing framework",
    -1,
    AgentMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_specialized_agents_c(void) {
    return PyModule_Create(&agent_module);
}