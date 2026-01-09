#ifndef UTIL_H
#define UTIL_H

#define VERSION 0

#ifdef debug
#define DEBUG(x) printf("VERSION: %d %s\n", VERSION, x)
#else
#define DEBUG(x)
#endif

#endif
