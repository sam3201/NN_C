#include <stdio.h>

#define STATUS(fn_call, status_var)      \
    do {                                 \
        int _r = (fn_call);              \
        if (_r == 0)                     \
            (status_var) = 1;            \
        else                             \
            (status_var) = 0;            \
    } while (0)

void hello(int argc, char **argv) {
    printf("hello world\n");
}

int main(void) {
    int status;

    STATUS(hello(0, NULL), status);

    printf("status = %d\n", status); // prints 1 if success

    return status;
}

