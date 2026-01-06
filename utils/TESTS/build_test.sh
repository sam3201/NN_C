(base) samueldasari@Samuels-MacBook-Air TESTS % clang -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I../NN -I../NN/MUZE -I../NN/MEMORY -I../NN/TOKENIZER -I../NN/TRANSFORMER \
  sam_muze_toy_test.c \
  ../NN/NEAT.c ../NN/NN.c ../NN/TOKENIZER.c ../NN/TRANSFORMER.c \
  -lm -o sam_muze_toy_test

../NN/TOKENIZER.c:1:10: warning: non-portable path to file '"TOKENIZER.h"'; specified path differs in case from file name on disk [-Wnonportable-include-path]
    1 | #include "tokenizer.h"
      |          ^~~~~~~~~~~~~
      |          "TOKENIZER.h"
1 warning generated.
Undefined symbols for architecture arm64:
  "_SAM_MUZE_destroy", referenced from:
      _main in sam_muze_toy_test-67fb17.o
  "_SAM_as_MUZE", referenced from:
      _main in sam_muze_toy_test-67fb17.o
  "_SAM_destroy", referenced from:
      _main in sam_muze_toy_test-67fb17.o
      _main in sam_muze_toy_test-67fb17.o
  "_SAM_init", referenced from:
      _main in sam_muze_toy_test-67fb17.o
  "_muze_plan", referenced from:
      _main in sam_muze_toy_test-67fb17.o
  "_toy_env_reset", referenced from:
      _main in sam_muze_toy_test-67fb17.o
  "_toy_env_step", referenced from:
      _main in sam_muze_toy_test-67fb17.o
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
(base) samueldasari@Samuels-MacBook-Air TESTS % clang -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I../../SAM -I../NN -I../NN/MUZE -I../NN/MEMORY -I../NN/TOKENIZER -I../NN/TRANSFORMER \
  sam_muze_toy_test.c \
  ../NN/NEAT.c ../NN/NN.c ../NN/TOKENIZER.c ../NN/TRANSFORMER.c \
  -lm -o sam_muze_toy_test

../NN/TOKENIZER.c:1:10: warning: non-portable path to file '"TOKENIZER.h"'; specified path differs in case from file name on disk [-Wnonportable-include-path]
    1 | #include "tokenizer.h"
      |          ^~~~~~~~~~~~~
      |          "TOKENIZER.h"
1 warning generated.
Undefined symbols for architecture arm64:
  "_SAM_MUZE_destroy", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
  "_SAM_as_MUZE", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
  "_SAM_destroy", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
      _main in sam_muze_toy_test-9cc58e.o
  "_SAM_init", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
  "_muze_plan", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
  "_toy_env_reset", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
  "_toy_env_step", referenced from:
      _main in sam_muze_toy_test-9cc58e.o
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
