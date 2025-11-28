#include <stdio.h>

#define STATUS _typeof(0)
  int: "%d", \
  char: "%c", \
  float: "%f", \
  double: "%f", \
  default: "%p")

int main(void) {
  printf("hello world\n");

  return 0;

}
