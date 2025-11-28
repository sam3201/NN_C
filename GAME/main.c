#include <stdio.h>

#ifdef ERROR(void*fn(int argc, char **argv)) 
  #define STATUS 0
#else
  #define STATUS 1
#endif

int hello(int argc, char **argv) {
  printf("hello world\n");

  
}

int main(void) {
    printf("hello world\n");

    return  
}

