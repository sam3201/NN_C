#include <stdio.h>

#ifdef ERROR
  #assert(0);
  
  #define STATUS 0
#else
  #define STATUS 1
#endif

int main(void) {
    printf("hello world\n");

    return STATUS;
}

