#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/* Visibility tokens used in api.def */
#define PUBLIC  1
#define PRIVATE 0

/* If 1, include PRIVATE too. Otherwise only PUBLIC. */
#ifndef API_VIS_PRIVATE_TOO
#define API_VIS_PRIVATE_TOO 0
#endif

/* If 1, only emit selected symbols (IMPORT_<name>=1). */
#ifndef API_SELECTIVE
#define API_SELECTIVE 0
#endif

/* ===== Default IMPORT_* macros to 0 in selective mode ===== */
#if API_SELECTIVE
  #define API_TYPE(vis, name, body) \
    #ifndef IMPORT_##name \
    #define IMPORT_##name 0 \
    #endif

  #define API_FN(vis, ret, name, sig) \
    #ifndef IMPORT_##name \
    #define IMPORT_##name 0 \
    #endif

  #include "api.def"
  #undef API_TYPE
  #undef API_FN
#endif

/* ===== Emit declarations ===== */
#if API_SELECTIVE

  #define API_TYPE(vis, name, body) \
    #if (API_VIS_PRIVATE_TOO || (vis)) \
      #if IMPORT_##name \
        typedef struct name { body } name; \
      #endif \
    #endif

  #define API_FN(vis, ret, name, sig) \
    #if (API_VIS_PRIVATE_TOO || (vis)) \
      #if IMPORT_##name \
        ret name sig; \
      #endif \
    #endif

#else

  #define API_TYPE(vis, name, body) \
    #if (API_VIS_PRIVATE_TOO || (vis)) \
      typedef struct name { body } name; \
    #endif

  #define API_FN(vis, ret, name, sig) \
    #if (API_VIS_PRIVATE_TOO || (vis)) \
      ret name sig; \
    #endif

#endif

#include "api.def"
#undef API_TYPE
#undef API_FN

#ifdef __cplusplus
}
#endif

