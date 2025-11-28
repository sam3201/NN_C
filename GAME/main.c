#include <stdio.h>

#ifdef STATUS(void*fn(int argc, char **argv)) 
  int res = assert(fn(argc, argv) == 0);
  #if 
  #define ERROR 0
#else
  #define SUCCESS 1
#endif

void hello(int argc, char **argv) {
  printf("hello world\n");
}

int main(void) {
 
}

