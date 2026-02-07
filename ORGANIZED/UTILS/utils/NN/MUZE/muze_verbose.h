#ifndef MUZE_VERBOSE_H
#define MUZE_VERBOSE_H

#include <stdio.h>

// Global verbose flag for MUZE print statements
extern int g_muze_verbose;

// Function to set verbose mode
void muze_set_verbose(int verbose);

// Function to get verbose mode
int muze_get_verbose(void);

// Macros for conditional printing
#define MUZE_PRINTF(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MUZE_PRINT_REANALYZE(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf("[reanalyze] " fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MUZE_PRINT_EVAL(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf("[eval] " fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MUZE_PRINT_LOOP(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf("[loop] " fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MUZE_PRINT_TRAIN(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf("[train " fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define MUZE_PRINT_SELFPLAY(fmt, ...) \
    do { \
        if (g_muze_verbose) { \
            printf("[selfplay] " fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#endif // MUZE_VERBOSE_H
