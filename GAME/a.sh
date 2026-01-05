# MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')
# SAM_SRC=$(find ../utils/NN/SAM -name "*.c" | tr '\n' ' ')

gcc -g game.c \
    -I../utils/Raylib/src \
    -I../utils/NN/MUZE \
    -I../utils/NN/SAM \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o game

