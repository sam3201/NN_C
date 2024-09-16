#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "CodeRunner.h"

#define MAX_TOKEN_LENGTH 64

Token *tokenize_text(char *text, int *token_count) {
    Token *tokens = NULL;
    *token_count = 0;

    char *token_str = strtok(text, " \n\t");
    while (token_str != NULL) {
        Token token;
        if (isalpha(token_str[0])) {
            // Check if it's an instruction or directive
            if (strcmp(token_str, "MOV") == 0 || strcmp(token_str, "ADD") == 0) {
                token.type = TOKEN_INSTRUCTION;
            } else if (token_str[0] == '.') {
                token.type = TOKEN_DIRECTIVE;
            } else {
                token.type = TOKEN_LABEL; // Assume label for anything else alphabetical
            }
        } else if (isdigit(token_str[0]) || token_str[0] == '0' && token_str[1] == 'x') {
            token.type = TOKEN_NUMBER;
        } else {
            token.type = TOKEN_UNKNOWN;
        }
        token.value = strdup(token_str);
        tokens = realloc(tokens, sizeof(Token) * (*token_count + 1));
        tokens[*token_count] = token;
        (*token_count)++;
        token_str = strtok(NULL, " \n\t");
    }
    return tokens;
}

void save_state(CodeRunner *runner, char *new_state) {
    runner->states[runner->state_count] = strdup(new_state);
    runner->state_count++;
}

void print_tokens(Token *tokens, int token_count) {
    for (int i = 0; i < token_count; i++) {
        printf("Token %d: Type %d, Value: %s\n", i, tokens[i].type, tokens[i].value);
    }
}

