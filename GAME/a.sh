# MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
# SAM_SRC=$(find ../utils/NN/SAM -name "*.c" | tr '\n' ' ')

gcc -g game.c \
    -I../utils/Raylib/src \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o game

./game

rm game
