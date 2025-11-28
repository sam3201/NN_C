#include <stdio.h>

#define STATUS _Generic(NULL, \
  int: "%d", \
  char: "%c", \
  float: "%f", \
  double: "%f", \
  default: "%p")

int main(void) {
  printf("hello world\n");

  return 0;

}
