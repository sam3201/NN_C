#!/bin/bash

# Compile the visualizer server
gcc -o vis_server utils/VISUALIZER/server.c utils/VISUALIZER/NN_visualizer.c utils/VISUALIZER/nn_protocol.c utils/VISUALIZER/client.c utils/NN/NN.c utils/NN/NEAT.c -I. -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lpthread -lm -framework OpenGL -framework Cocoa -framework IOKit -O2
if [ $? -ne 0 ]; then
    echo "Visualizer server compilation failed!"
    exit 1
fi

# Compile the main game
gcc -o main main.c utils/VISUALIZER/NN_visualizer.c utils/VISUALIZER/nn_protocol.c utils/VISUALIZER/client.c utils/NN/NN.c utils/NN/NEAT.c -I. -I/opt/homebrew/include -L/opt/homebrew/lib -lraylib -lpthread -lm -framework OpenGL -framework Cocoa -framework IOKit -O2
if [ $? -ne 0 ]; then
    echo "Main game compilation failed!"
    exit 1
fi

echo "Compilation successful! Starting the visualizer server..."

# Run the visualizer server in the background
./vis_server &
server_pid=$!

# Wait a moment for the server to start
sleep 1

echo "Starting the main game..."
# Run the main game
./main

# Cleanup
echo "Cleaning up..."
kill $server_pid
wait $server_pid 2>/dev/null
rm vis_server main

echo "Done!"
