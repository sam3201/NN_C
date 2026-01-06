MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
NN_SRC=$(find ../utils/NN -name "*.c" | tr '\n' ' ')
SAM_SRC=$(find ../SAM -name "*.c" | tr '\n' ' ')

gcc -w game.c $NN_SRC $MUZE_SRC $SAM_SRC \
  -I../utils/NN -I../utils/NN/MUZE -I../SAM -I../utils/Raylib/src \
  -L../utils/Raylib/src -lraylib \
  -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
  -arch arm64 \
  -o game


./game

rm game
