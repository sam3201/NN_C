typedef enum {
  ERROR,
  WARNING,
  INFO,
  DEBUG,
} ConsoleError; 

typedef enum {
  TOKEN_INSTRUCTION, 
  TOKEN_REGISTER,   
  TOKEN_NUMBER,    
  TOKEN_LABEL,    
  TOKEN_DIRECTIVE,   
  TOKEN_UNKNOWN,    
} TokenType;

typedef struct {
  TokenType type;
  char *value;
} Token;

typedef struct {
  char *text;
  Token *tokens; 
  int token_count;
  int state_count;
  char *states[]; 
} CodeRunner;

CodeRunner *CodeRunner_new(char *text);

unsigned char text_to_hex(char *text);

Token *tokenize_text(char *text, int *token_count);
void save_state(CodeRunner *runner, char *new_state);
void print_tokens(Token *tokens, int token_count);

