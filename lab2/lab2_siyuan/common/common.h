#ifndef __MY_COMMON_H
#define __MY_COMMON_H

#include <cstdio>
#include <cstdlib>

#ifdef VERBOSE
#   define MYDEBUG(...) fprintf(stdout, __VA_ARGS__)
#else
#   define MYDEBUG(...) 
#endif

#endif
