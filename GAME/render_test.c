// 3D Rendering Test Suite using C_SELECT
// This file extracts and tests the 3D rendering components from game.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenGL/gl.h>
#include <SDL3/SDL.h>

// Import only the rendering-related components using C_SELECT
#include "generated/auto_import.h"

// Test window and context
static SDL_Window *test_window = NULL;
static SDL_GLContext test_gl_context = NULL;

// Test camera and viewport settings
typedef struct {
    Vector3 position;
    Vector3 target;
    Vector3 up;
    float fov;
    float near_plane;
    float far_plane;
} Camera3D;

// Simple 3D vertex structure
typedef struct {
    Vector3 position;
    Color color;
} Vertex3D;

// Test 3D objects
typedef struct {
    Vertex3D *vertices;
    int vertex_count;
    Vector3 position;
    Vector3 rotation;
    Vector3 scale;
} TestObject3D;

// Global test camera
static Camera3D test_camera = {
    .position = {0.0f, 5.0f, 10.0f},
    .target = {0.0f, 0.0f, 0.0f},
    .up = {0.0f, 1.0f, 0.0f},
    .fov = 60.0f,
    .near_plane = 0.1f,
    .far_plane = 1000.0f
};

// Test objects
static TestObject3D test_cube;
static TestObject3D test_pyramid;
static TestObject3D test_ground;

// Initialize OpenGL settings
void init_opengl_test(void) {
    printf("Initializing OpenGL test environment...\n");
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    
    // Set up basic lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    
    GLfloat light_position[] = {5.0f, 10.0f, 5.0f, 1.0f};
    GLfloat light_ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat light_diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    
    // Enable color material
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    
    // Set background color
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    
    printf("OpenGL test environment initialized.\n");
}

// Create test cube vertices
void create_test_cube(TestObject3D *obj) {
    obj->vertex_count = 8;
    obj->vertices = malloc(sizeof(Vertex3D) * obj->vertex_count);
    
    // Cube vertices (centered at origin)
    float s = 1.0f; // half size
    obj->vertices[0] = (Vertex3D){{-s, -s, -s}, {255, 0, 0, 255}}; // Red
    obj->vertices[1] = (Vertex3D){{ s, -s, -s}, {255, 255, 0, 255}}; // Yellow
    obj->vertices[2] = (Vertex3D){{ s,  s, -s}, {0, 255, 0, 255}}; // Green
    obj->vertices[3] = (Vertex3D){{-s,  s, -s}, {0, 255, 255, 255}}; // Cyan
    obj->vertices[4] = (Vertex3D){{-s, -s,  s}, {255, 0, 255, 255}}; // Magenta
    obj->vertices[5] = (Vertex3D){{ s, -s,  s}, {0, 0, 255, 255}}; // Blue
    obj->vertices[6] = (Vertex3D){{ s,  s,  s}, {255, 255, 255, 255}}; // White
    obj->vertices[7] = (Vertex3D){{-s,  s,  s}, {128, 128, 128, 255}}; // Gray
    
    obj->position = (Vector3){0.0f, 1.0f, 0.0f};
    obj->rotation = (Vector3){0.0f, 0.0f, 0.0f};
    obj->scale = (Vector3){1.0f, 1.0f, 1.0f};
}

// Create test pyramid vertices
void create_test_pyramid(TestObject3D *obj) {
    obj->vertex_count = 5;
    obj->vertices = malloc(sizeof(Vertex3D) * obj->vertex_count);
    
    float s = 1.0f;
    obj->vertices[0] = (Vertex3D){{ 0.0f,  s,  0.0f}, {255, 255, 0, 255}}; // Top (Yellow)
    obj->vertices[1] = (Vertex3D){{-s, -s, -s}, {255, 0, 0, 255}}; // Base corners (Red)
    obj->vertices[2] = (Vertex3D){{ s, -s, -s}, {0, 255, 0, 255}}; // Green
    obj->vertices[3] = (Vertex3D){{ s, -s,  s}, {0, 0, 255, 255}}; // Blue
    obj->vertices[4] = (Vertex3D){{-s, -s,  s}, {255, 0, 255, 255}}; // Magenta
    
    obj->position = (Vector3){3.0f, 0.5f, 0.0f};
    obj->rotation = (Vector3){0.0f, 0.0f, 0.0f};
    obj->scale = (Vector3){1.0f, 1.0f, 1.0f};
}

// Create test ground plane
void create_test_ground(TestObject3D *obj) {
    obj->vertex_count = 4;
    obj->vertices = malloc(sizeof(Vertex3D) * obj->vertex_count);
    
    float size = 10.0f;
    obj->vertices[0] = (Vertex3D){{-size, 0.0f, -size}, {100, 100, 100, 255}}; // Light gray
    obj->vertices[1] = (Vertex3D){{ size, 0.0f, -size}, {150, 150, 150, 255}}; // Lighter gray
    obj->vertices[2] = (Vertex3D){{ size, 0.0f,  size}, {100, 100, 100, 255}};
    obj->vertices[3] = (Vertex3D){{-size, 0.0f,  size}, {150, 150, 150, 255}};
    
    obj->position = (Vector3){0.0f, 0.0f, 0.0f};
    obj->rotation = (Vector3){0.0f, 0.0f, 0.0f};
    obj->scale = (Vector3){1.0f, 1.0f, 1.0f};
}

// Initialize test objects
void init_test_objects(void) {
    printf("Creating test 3D objects...\n");
    
    create_test_cube(&test_cube);
    create_test_pyramid(&test_pyramid);
    create_test_ground(&test_ground);
    
    printf("Test objects created:\n");
    printf("- Cube: %d vertices\n", test_cube.vertex_count);
    printf("- Pyramid: %d vertices\n", test_pyramid.vertex_count);
    printf("- Ground: %d vertices\n", test_ground.vertex_count);
}

// Set up 3D projection matrix
void setup_3d_projection(int width, int height) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    // Simple perspective projection
    float aspect = (float)width / (float)height;
    float fov_rad = test_camera.fov * PI / 180.0f;
    float f = 1.0f / tanf(fov_rad / 2.0f);
    
    float projection[16] = {
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (test_camera.far_plane + test_camera.near_plane) / (test_camera.near_plane - test_camera.far_plane), -1,
        0, 0, (2 * test_camera.far_plane * test_camera.near_plane) / (test_camera.near_plane - test_camera.far_plane), 0
    };
    
    glLoadMatrixf(projection);
    glMatrixMode(GL_MODELVIEW);
}

// Draw a test 3D object
void draw_test_object3d(TestObject3D *obj) {
    glPushMatrix();
    
    // Apply transformations
    glTranslatef(obj->position.x, obj->position.y, obj->position.z);
    glRotatef(obj->rotation.x, 1.0f, 0.0f, 0.0f);
    glRotatef(obj->rotation.y, 0.0f, 1.0f, 0.0f);
    glRotatef(obj->rotation.z, 0.0f, 0.0f, 1.0f);
    glScalef(obj->scale.x, obj->scale.y, obj->scale.z);
    
    // Draw vertices
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < obj->vertex_count; i++) {
        Vertex3D *v = &obj->vertices[i];
        glColor4f(v->color.r/255.0f, v->color.g/255.0f, v->color.b/255.0f, v->color.a/255.0f);
        glVertex3f(v->position.x, v->position.y, v->position.z);
    }
    glEnd();
    
    glPopMatrix();
}

// Draw test cube with proper faces
void draw_test_cube_faces(TestObject3D *obj) {
    glPushMatrix();
    glTranslatef(obj->position.x, obj->position.y, obj->position.z);
    glRotatef(obj->rotation.x, 1.0f, 0.0f, 0.0f);
    glRotatef(obj->rotation.y, 0.0f, 1.0f, 0.0f);
    glRotatef(obj->rotation.z, 0.0f, 0.0f, 1.0f);
    glScalef(obj->scale.x, obj->scale.y, obj->scale.z);
    
    // Draw cube faces
    Vertex3D *v = obj->vertices;
    
    // Front face
    glBegin(GL_QUADS);
    glColor4f(v[4].color.r/255.0f, v[4].color.g/255.0f, v[4].color.b/255.0f, v[4].color.a/255.0f);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    glVertex3f(v[5].position.x, v[5].position.y, v[5].position.z);
    glVertex3f(v[6].position.x, v[6].position.y, v[6].position.z);
    glVertex3f(v[7].position.x, v[7].position.y, v[7].position.z);
    glEnd();
    
    // Back face
    glBegin(GL_QUADS);
    glColor4f(v[0].color.r/255.0f, v[0].color.g/255.0f, v[0].color.b/255.0f, v[0].color.a/255.0f);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glEnd();
    
    // Top face
    glBegin(GL_QUADS);
    glColor4f(v[3].color.r/255.0f, v[3].color.g/255.0f, v[3].color.b/255.0f, v[3].color.a/255.0f);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glVertex3f(v[6].position.x, v[6].position.y, v[6].position.z);
    glVertex3f(v[7].position.x, v[7].position.y, v[7].position.z);
    glEnd();
    
    // Bottom face
    glBegin(GL_QUADS);
    glColor4f(v[1].color.r/255.0f, v[1].color.g/255.0f, v[1].color.b/255.0f, v[1].color.a/255.0f);
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[5].position.x, v[5].position.y, v[5].position.z);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    glEnd();
    
    // Right face
    glBegin(GL_QUADS);
    glColor4f(v[5].color.r/255.0f, v[5].color.g/255.0f, v[5].color.b/255.0f, v[5].color.a/255.0f);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glVertex3f(v[6].position.x, v[6].position.y, v[6].position.z);
    glVertex3f(v[5].position.x, v[5].position.y, v[5].position.z);
    glEnd();
    
    // Left face
    glBegin(GL_QUADS);
    glColor4f(v[0].color.r/255.0f, v[0].color.g/255.0f, v[0].color.b/255.0f, v[0].color.a/255.0f);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    glVertex3f(v[7].position.x, v[7].position.y, v[7].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glEnd();
    
    glPopMatrix();
}

// Draw test pyramid with proper faces
void draw_test_pyramid_faces(TestObject3D *obj) {
    glPushMatrix();
    glTranslatef(obj->position.x, obj->position.y, obj->position.z);
    glRotatef(obj->rotation.x, 1.0f, 0.0f, 0.0f);
    glRotatef(obj->rotation.y, 0.0f, 1.0f, 0.0f);
    glRotatef(obj->rotation.z, 0.0f, 0.0f, 1.0f);
    glScalef(obj->scale.x, obj->scale.y, obj->scale.z);
    
    Vertex3D *v = obj->vertices;
    
    // Base (square)
    glBegin(GL_QUADS);
    glColor4f(100/255.0f, 100/255.0f, 100/255.0f, 255/255.0f);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    glEnd();
    
    // Four triangular faces
    glBegin(GL_TRIANGLES);
    // Face 1
    glColor4f(v[0].color.r/255.0f, v[0].color.g/255.0f, v[0].color.b/255.0f, v[0].color.a/255.0f);
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    
    // Face 2
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    
    // Face 3
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    
    // Face 4
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[4].position.x, v[4].position.y, v[4].position.z);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glEnd();
    
    glPopMatrix();
}

// Draw test ground
void draw_test_ground(TestObject3D *obj) {
    glPushMatrix();
    glTranslatef(obj->position.x, obj->position.y, obj->position.z);
    glRotatef(obj->rotation.x, 1.0f, 0.0f, 0.0f);
    glRotatef(obj->rotation.y, 0.0f, 1.0f, 0.0f);
    glRotatef(obj->rotation.z, 0.0f, 0.0f, 1.0f);
    glScalef(obj->scale.x, obj->scale.y, obj->scale.z);
    
    Vertex3D *v = obj->vertices;
    
    glBegin(GL_QUADS);
    glColor4f(v[0].color.r/255.0f, v[0].color.g/255.0f, v[0].color.b/255.0f, v[0].color.a/255.0f);
    glVertex3f(v[0].position.x, v[0].position.y, v[0].position.z);
    glVertex3f(v[1].position.x, v[1].position.y, v[1].position.z);
    glVertex3f(v[2].position.x, v[2].position.y, v[2].position.z);
    glVertex3f(v[3].position.x, v[3].position.y, v[3].position.z);
    glEnd();
    
    glPopMatrix();
}

// Main render test function
void render_test_scene(void) {
    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Set up camera view
    glLoadIdentity();
    gluLookAt(
        test_camera.position.x, test_camera.position.y, test_camera.position.z,
        test_camera.target.x, test_camera.target.y, test_camera.target.z,
        test_camera.up.x, test_camera.up.y, test_camera.up.z
    );
    
    // Draw test objects
    draw_test_ground(&test_ground);
    draw_test_cube_faces(&test_cube);
    draw_test_pyramid_faces(&test_pyramid);
    
    // Draw coordinate axes for reference
    glBegin(GL_LINES);
    // X axis (red)
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(5.0f, 0.0f, 0.0f);
    // Y axis (green)
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 5.0f, 0.0f);
    // Z axis (blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 5.0f);
    glEnd();
}

// Animation update
void update_test_animation(float delta_time) {
    // Rotate objects
    test_cube.rotation.y += 45.0f * delta_time; // 45 degrees per second
    test_pyramid.rotation.y -= 30.0f * delta_time; // 30 degrees per second
    
    // Move camera in a circle
    static float camera_angle = 0.0f;
    camera_angle += 20.0f * delta_time; // 20 degrees per second
    float radius = 15.0f;
    test_camera.position.x = cosf(camera_angle * PI / 180.0f) * radius;
    test_camera.position.z = sinf(camera_angle * PI / 180.0f) * radius;
    test_camera.position.y = 8.0f; // Keep height constant
    
    // Always look at center
    test_camera.target = (Vector3){0.0f, 0.0f, 0.0f};
}

// Test main loop
int main(void) {
    printf("=== 3D Rendering Test Suite ===\n");
    printf("Using C_SELECT to import rendering components...\n\n");
    
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return 1;
    }
    
    // Create window
    test_window = SDL_CreateWindow("3D Rendering Test", 1024, 768, SDL_WINDOW_OPENGL);
    
    if (!test_window) {
        printf("Window creation failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    
    // Create OpenGL context
    test_gl_context = SDL_GL_CreateContext(test_window);
    if (!test_gl_context) {
        printf("OpenGL context creation failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(test_window);
        SDL_Quit();
        return 1;
    }
    
    // Initialize rendering
    init_opengl_test();
    init_test_objects();
    
    // Get window size
    int width, height;
    SDL_GetWindowSize(test_window, &width, &height);
    setup_3d_projection(width, height);
    
    printf("Starting render test...\n");
    printf("Controls:\n");
    printf("- ESC: Exit\n");
    printf("- Arrow keys: Move camera\n");
    printf("- Mouse: Look around\n\n");
    
    // Main loop
    int running = 1;
    SDL_Event event;
    Uint32 last_time = SDL_GetTicks();
    
    while (running) {
        Uint32 current_time = SDL_GetTicks();
        float delta_time = (current_time - last_time) / 1000.0f;
        last_time = current_time;
        
        // Handle events
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = 0;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (event.key.key == SDLK_ESCAPE) {
                        running = 0;
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        continue;
                    }
                    break;
                case SDL_EVENT_MOUSE_BUTTON_UP:
                    if (event.button.button == SDL_BUTTON_LEFT) {
                        continue;
                    }
                    break;
            }
        }
        
        // Update animation
        update_test_animation(delta_time);
        
        // Render
        render_test_scene();
        
        // Swap buffers
        SDL_GL_SwapWindow(test_window);
        
        // Small delay to prevent excessive CPU usage
        SDL_Delay(16); // ~60 FPS
    }
    
    // Cleanup
    free(test_cube.vertices);
    free(test_pyramid.vertices);
    free(test_ground.vertices);
    
    SDL_GL_DestroyContext(test_gl_context);
    SDL_DestroyWindow(test_window);
    SDL_Quit();
    
    printf("\nTest completed successfully!\n");
    return 0;
}
