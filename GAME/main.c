#include <stdio.h>

#ifdef STATUS(void*fn(int argc, char **argv)) 
  #define ERROR 0
#else
  #define SUCCESS 1
#endif

void hello(int argc, char **argv) {
  printf("hello world\n");
}

int main(void) {
 
}

