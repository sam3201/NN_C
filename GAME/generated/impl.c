#include <stdarg.h>
#include <math.h>
#include <OpenGL/gl.h>
#include <stdio.h>
#include <SDL3/SDL.h>

// Include type definitions
#include "complete_api.h"

// External declaration for the SDL window
extern SDL_Window *g_window;

// OpenGL Drawing Function Implementations (these are unique to our system)
static void draw_rectangle_outline_opengl(int x, int y, int width, int height, Color color) {
    glBegin(GL_LINE_LOOP);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();
}

// Simple gluLookAt implementation for 3D camera setup
static void gluLookAt(float eyeX, float eyeY, float eyeZ, 
                    float centerX, float centerY, float centerZ,
                    float upX, float upY, float upZ) {
    // Calculate forward vector
    float forwardX = centerX - eyeX;
    float forwardY = centerY - eyeY;
    float forwardZ = centerZ - eyeZ;
    
    // Normalize forward vector
    float length = sqrtf(forwardX * forwardX + forwardY * forwardY + forwardZ * forwardZ);
    if (length > 0.0f) {
        forwardX /= length;
        forwardY /= length;
        forwardZ /= length;
    }
    
    // Calculate right vector (cross product of forward and up)
    float rightX = forwardY * upZ - forwardZ * upY;
    float rightY = forwardZ * upX - forwardX * upZ;
    float rightZ = forwardX * upY - forwardY * upX;
    
    // Normalize right vector
    length = sqrtf(rightX * rightX + rightY * rightY + rightZ * rightZ);
    if (length > 0.0f) {
        rightX /= length;
        rightY /= length;
        rightZ /= length;
    }
    
    // Calculate true up vector (cross product of right and forward)
    float trueUpX = rightY * forwardZ - rightZ * forwardY;
    float trueUpY = rightZ * forwardX - rightX * forwardZ;
    float trueUpZ = rightX * forwardY - rightY * forwardX;
    
    // Create view matrix
    float matrix[16] = {
        rightX, trueUpX, -forwardX, 0,
        rightY, trueUpY, -forwardY, 0,
        rightZ, trueUpZ, -forwardZ, 0,
        -(rightX * eyeX + rightY * eyeY + rightZ * eyeZ),
        -(trueUpX * eyeX + trueUpY * eyeY + trueUpZ * eyeZ),
        (forwardX * eyeX + forwardY * eyeY + forwardZ * eyeZ), 1
    };
    
    glMultMatrixf(matrix);
}


