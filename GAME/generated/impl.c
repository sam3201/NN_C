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
// Note: draw_rectangle_outline_opengl is called by the game but not implemented there
void draw_rectangle_outline_opengl(int x, int y, int width, int height, Color color) {
    glBegin(GL_LINE_LOOP);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();
}
