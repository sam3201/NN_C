#include <stdio.h>

#define STATUS \
 #ifdef DEBUG \
  "debug" \
 #else \
  "release" \
 #endif

int main(void) {
  printf("hello world\n");

  return 0;

}
