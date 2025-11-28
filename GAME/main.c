#include <stdio.h>

#ifdef ERROR
  //check if it breaks! Then return the status!
  #assert(0);
  
  #define STATUS 0
#else
  #define STATUS 1
#endif

int main(void) {
    printf("hello world\n");

    return STATUS;
}

