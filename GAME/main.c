#include <stdio.h>

#ifdef ERROR
  #check if it breaks 
  #assert(0);
  #define STATUS 0
#else
  #define STATUS 1
#endif

int main(void) {
    printf("hello world\n");

    return STATUS;
}

