# SDL_RenderGeometry

Render a list of triangles, optionally using a texture and indices into the vertex array Color and alpha modulation is done per vertex ([SDL_SetTextureColorMod](SDL_SetTextureColorMod) and [SDL_SetTextureAlphaMod](SDL_SetTextureAlphaMod) are ignored).

## Header File

Defined in [<SDL3/SDL_render.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_render.h)

## Syntax

```c
bool SDL_RenderGeometry(SDL_Renderer *renderer,
                   SDL_Texture *texture,
                   const SDL_Vertex *vertices, int num_vertices,
                   const int *indices, int num_indices);
```

## Function Parameters

|                                  |                  |                                                                                                                              |
| -------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [SDL_Renderer](SDL_Renderer) *   | **renderer**     | the rendering context.                                                                                                       |
| [SDL_Texture](SDL_Texture) *     | **texture**      | (optional) The SDL texture to use.                                                                                           |
| const [SDL_Vertex](SDL_Vertex) * | **vertices**     | vertices.                                                                                                                    |
| int                              | **num_vertices** | number of vertices.                                                                                                          |
| const int *                      | **indices**      | (optional) An array of integer indices into the 'vertices' array, if NULL all vertices will be rendered in sequential order. |
| int                              | **num_indices**  | number of indices.                                                                                                           |

## Return Value

(bool) Returns true on success or false on failure; call
[SDL_GetError](SDL_GetError)() for more information.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_RenderGeometryRaw](SDL_RenderGeometryRaw)
- [SDL_SetRenderTextureAddressMode](SDL_SetRenderTextureAddressMode)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryRender](CategoryRender)

