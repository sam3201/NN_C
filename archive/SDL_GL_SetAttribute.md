# SDL_GL_SetAttribute

Set an OpenGL window attribute before window creation.

## Header File

Defined in [<SDL3/SDL_video.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_video.h)

## Syntax

```c
bool SDL_GL_SetAttribute(SDL_GLAttr attr, int value);
```

## Function Parameters

|                          |           |                                                       |
| ------------------------ | --------- | ----------------------------------------------------- |
| [SDL_GLAttr](SDL_GLAttr) | **attr**  | an enum value specifying the OpenGL attribute to set. |
| int                      | **value** | the desired value for the attribute.                  |

## Return Value

(bool) Returns true on success or false on failure; call
[SDL_GetError](SDL_GetError)() for more information.

## Remarks

This function sets the OpenGL attribute `attr` to `value`. The requested
attributes should be set before creating an OpenGL window. You should use
[SDL_GL_GetAttribute](SDL_GL_GetAttribute)() to check the values after
creating the OpenGL context, since the values obtained can differ from the
requested ones.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_GL_CreateContext](SDL_GL_CreateContext)
- [SDL_GL_GetAttribute](SDL_GL_GetAttribute)
- [SDL_GL_ResetAttributes](SDL_GL_ResetAttributes)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryVideo](CategoryVideo)

