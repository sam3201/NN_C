MUZE_SRC=$(find ../utils/NN/MUZE -name "*.c" | tr '\n' ' ')

clang -fsanitize=address -g -O1 game.c $MUZE_SRC \
    -I../utils/Raylib/src \
    -I../utils/NN/MUZE \
    -L../utils/Raylib/src -lraylib \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo \
    -o game

