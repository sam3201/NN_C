#include "muze_verbose.h"

// Global verbose flag initialization
int g_muze_verbose = 0; // Default to quiet (0), set to 1 to enable verbose

void muze_set_verbose(int verbose) {
    g_muze_verbose = verbose;
}

int muze_get_verbose(void) {
    return g_muze_verbose;
}
