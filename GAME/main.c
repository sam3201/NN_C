#include <stdio.h>

#ifdef STATUS(void*fn(int argc, char **argv)) 
  int res = assert(fn(argc, argv) == 0);
  #if res
    #define SUCCESS 1
  #else
    #define ERROR 0
  #endif
#endif

void hello(int argc, char **argv) {
  printf("hello world\n");
}

int main(void) {
  int status; 
  STATUS(hello(0, NULL));

  return status;
}

