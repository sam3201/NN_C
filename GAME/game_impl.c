#include <stdarg.h>
#include <math.h>
#include <OpenGL/gl.h>
#include "../utils/Raylib/src/raylib.h"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

// OpenGL Drawing Function Implementations
void draw_circle_filled_opengl(float centerX, float centerY, float radius, Color color) {
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(centerX, centerY);
    int segments = 32;
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * PI * (float)i / (float)segments;
        glVertex2f(centerX + cosf(angle) * radius, centerY + sinf(angle) * radius);
    }
    glEnd();
}

void draw_circle_outline_opengl(float centerX, float centerY, float radius, Color color) {
    glBegin(GL_LINE_LOOP);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    int segments = 32;
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * PI * (float)i / (float)segments;
        glVertex2f(centerX + cosf(angle) * radius, centerY + sinf(angle) * radius);
    }
    glEnd();
}

void draw_triangle_opengl(Vector2 v1, Vector2 v2, Vector2 v3, Color color) {
    glBegin(GL_TRIANGLES);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(v1.x, v1.y);
    glVertex2f(v2.x, v2.y);
    glVertex2f(v3.x, v3.y);
    glEnd();
}

void draw_line_opengl(float x1, float y1, float x2, float y2, float thickness, Color color) {
    glLineWidth(thickness);
    glBegin(GL_LINES);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(x1, y1);
    glVertex2f(x2, y2);
    glEnd();
    glLineWidth(1.0f); // Reset to default
}

void draw_ellipse_filled_opengl(int centerX, int centerY, int radiusX, int radiusY, Color color) {
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(centerX, centerY);
    int segments = 32;
    for (int i = 0; i <= segments; i++) {
        float angle = 2.0f * PI * (float)i / (float)segments;
        glVertex2f(centerX + cosf(angle) * radiusX, centerY + sinf(angle) * radiusY);
    }
    glEnd();
}

void draw_polygon_filled_opengl(Vector2 center, int sides, float radius, float rotation, Color color) {
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(center.x, center.y);
    for (int i = 0; i <= sides; i++) {
        float angle = rotation + 2.0f * PI * (float)i / (float)sides;
        glVertex2f(center.x + cosf(angle) * radius, center.y + sinf(angle) * radius);
    }
    glEnd();
}

void draw_polygon_outline_opengl(Vector2 center, int sides, float radius, float rotation, Color color) {
    glBegin(GL_LINE_LOOP);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    for (int i = 0; i <= sides; i++) {
        float angle = rotation + 2.0f * PI * (float)i / (float)sides;
        glVertex2f(center.x + cosf(angle) * radius, center.y + sinf(angle) * radius);
    }
    glEnd();
}

void draw_rectangle_outline_opengl(int x, int y, int width, int height, Color color) {
    glBegin(GL_LINE_LOOP);
    glColor4f(color.r/255.0f, color.g/255.0f, color.b/255.0f, color.a/255.0f);
    glVertex2f(x, y);
    glVertex2f(x + width, y);
    glVertex2f(x + width, y + height);
    glVertex2f(x, y + height);
    glEnd();
}
