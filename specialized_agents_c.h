/*
 * Pure C Specialized Agents Header
 * Research, Code Writing, Financial, Survival, and Meta agents
 */

#ifndef SPECIALIZED_AGENTS_C_H
#define SPECIALIZED_AGENTS_C_H

#include <stddef.h>

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
    double credibility_score;
} ResearcherAgent;

typedef struct {
    AgentBase base;
    void *code_transformer; // Transformer_t
    char **generated_code;
    size_t code_count;
    size_t code_capacity;
    double code_quality_score;
} CodeWriterAgent;

typedef struct {
    AgentBase base;
    void *market_model; // NEAT_t
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
    void *analysis_transformer; // Transformer_t
    char **code_improvements;
    size_t improvement_count;
    size_t improvement_capacity;
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

// ================================
// AGENT API FUNCTIONS
// ================================

// Create individual agents
ResearcherAgent *research_agent_create();
CodeWriterAgent *code_writer_agent_create();
FinancialAgent *financial_agent_create();
SurvivalAgent *survival_agent_create();
MetaAgent *meta_agent_create();

// Free individual agents
void research_agent_free(ResearcherAgent *agent);
void code_writer_agent_free(CodeWriterAgent *agent);
void financial_agent_free(FinancialAgent *agent);
void survival_agent_free(SurvivalAgent *agent);
void meta_agent_free(MetaAgent *agent);

// Agent execution functions
char *research_agent_perform_search(ResearcherAgent *agent, const char *query);
char *code_writer_agent_generate_code(CodeWriterAgent *agent, const char *spec);
char *financial_agent_analyze_market(FinancialAgent *agent, const char *market_data);
char *survival_agent_assess_threats(SurvivalAgent *agent);
char *meta_agent_analyze_system(MetaAgent *agent, const char *component);

// Agent registry functions
AgentRegistry *agent_registry_create();
void agent_registry_free(AgentRegistry *registry);

// Create all agents
int create_agents();

#endif // SPECIALIZED_AGENTS_C_H
