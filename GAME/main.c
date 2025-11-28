#include <stdio.h>

#ifdef ERROR
  #define STATUS 0
#else
  #define STATUS 1
#endif

int main(void) {
    printf("hello world\n");
    return STATUS;
}

