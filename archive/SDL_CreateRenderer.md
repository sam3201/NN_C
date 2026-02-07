# SDL_CreateRenderer

Create a 2D rendering context for a window.

## Header File

Defined in [<SDL3/SDL_render.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_render.h)

## Syntax

```c
SDL_Renderer * SDL_CreateRenderer(SDL_Window *window, const char *name);
```

## Function Parameters

|                            |            |                                                                                |
| -------------------------- | ---------- | ------------------------------------------------------------------------------ |
| [SDL_Window](SDL_Window) * | **window** | the window where rendering is displayed.                                       |
| const char *               | **name**   | the name of the rendering driver to initialize, or NULL to let SDL choose one. |

## Return Value

([SDL_Renderer](SDL_Renderer) *) Returns a valid rendering context or NULL
if there was an error; call [SDL_GetError](SDL_GetError)() for more
information.

## Remarks

If you want a specific renderer, you can specify its name here. A list of
available renderers can be obtained by calling
[SDL_GetRenderDriver](SDL_GetRenderDriver)() multiple times, with indices
from 0 to [SDL_GetNumRenderDrivers](SDL_GetNumRenderDrivers)()-1. If you
don't need a specific renderer, specify NULL and SDL will attempt to choose
the best option for you, based on what is available on the user's system.

If `name` is a comma-separated list, SDL will try each name, in the order
listed, until one succeeds or all of them fail.

By default the rendering size matches the window size in pixels, but you
can call
[SDL_SetRenderLogicalPresentation](SDL_SetRenderLogicalPresentation)() to
change the content size and scaling options.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_CreateRendererWithProperties](SDL_CreateRendererWithProperties)
- [SDL_CreateSoftwareRenderer](SDL_CreateSoftwareRenderer)
- [SDL_DestroyRenderer](SDL_DestroyRenderer)
- [SDL_GetNumRenderDrivers](SDL_GetNumRenderDrivers)
- [SDL_GetRenderDriver](SDL_GetRenderDriver)
- [SDL_GetRendererName](SDL_GetRendererName)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryRender](CategoryRender)

