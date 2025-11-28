#include <stdio.h>

#define STATUS \
 #ifdef ERROR \
  0 \
 #else \
  1 \
 #endif

int main(void) {
  printf("hello world\n");

  return STATUS; 

}
