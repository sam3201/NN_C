# SDL_CreateColorCursor

Create a color cursor.

## Header File

Defined in [<SDL3/SDL_mouse.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_mouse.h)

## Syntax

```c
SDL_Cursor * SDL_CreateColorCursor(SDL_Surface *surface,
                              int hot_x,
                              int hot_y);
```

## Function Parameters

|                              |             |                                                                        |
| ---------------------------- | ----------- | ---------------------------------------------------------------------- |
| [SDL_Surface](SDL_Surface) * | **surface** | an [SDL_Surface](SDL_Surface) structure representing the cursor image. |
| int                          | **hot_x**   | the x position of the cursor hot spot.                                 |
| int                          | **hot_y**   | the y position of the cursor hot spot.                                 |

## Return Value

([SDL_Cursor](SDL_Cursor) *) Returns the new cursor on success or NULL on
failure; call [SDL_GetError](SDL_GetError)() for more information.

## Remarks

If this function is passed a surface with alternate representations added
with [SDL_AddSurfaceAlternateImage](SDL_AddSurfaceAlternateImage)(), the
surface will be interpreted as the content to be used for 100% display
scale, and the alternate representations will be used for high DPI
situations if
[SDL_HINT_MOUSE_DPI_SCALE_CURSORS](SDL_HINT_MOUSE_DPI_SCALE_CURSORS) is
enabled. For example, if the original surface is 32x32, then on a 2x macOS
display or 200% display scale on Windows, a 64x64 version of the image will
be used, if available. If a matching version of the image isn't available,
the closest larger size image will be downscaled to the appropriate size
and be used instead, if available. Otherwise, the closest smaller image
will be upscaled and be used instead.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_AddSurfaceAlternateImage](SDL_AddSurfaceAlternateImage)
- [SDL_CreateAnimatedCursor](SDL_CreateAnimatedCursor)
- [SDL_CreateCursor](SDL_CreateCursor)
- [SDL_CreateSystemCursor](SDL_CreateSystemCursor)
- [SDL_DestroyCursor](SDL_DestroyCursor)
- [SDL_SetCursor](SDL_SetCursor)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryMouse](CategoryMouse)

