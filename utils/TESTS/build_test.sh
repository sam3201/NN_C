$MUZE_SRC (find ../NN/MUZE -name "*.c" -print)
$NN_SRC (find . ../NN/ -name "*.c" -print)
$SAM_SRC (find . -name "*.c" -print)

gcc -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I../../SAM -I../NN -I../NN/MUZE -I../NN/MEMORY -I../NN/TOKENIZER -I../NN/TRANSFORMER \
  sam_muze_toy_test.c \
  ../NN/NEAT.c ../NN/NN.c ../NN/TOKENIZER.c ../NN/TRANSFORMER.c ../../SAM/SAM.c ../../SAM/SAM_MUZE_ADAPTER.c \
  -lm -o sam_muze_toy_test
