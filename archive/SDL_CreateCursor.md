# SDL_CreateCursor

Create a cursor using the specified bitmap data and mask (in MSB format).

## Header File

Defined in [<SDL3/SDL_mouse.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_mouse.h)

## Syntax

```c
SDL_Cursor * SDL_CreateCursor(const Uint8 *data,
                         const Uint8 *mask,
                         int w, int h, int hot_x,
                         int hot_y);
```

## Function Parameters

|               |           |                                                                                                            |
| ------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| const Uint8 * | **data**  | the color value for each pixel of the cursor.                                                              |
| const Uint8 * | **mask**  | the mask value for each pixel of the cursor.                                                               |
| int           | **w**     | the width of the cursor.                                                                                   |
| int           | **h**     | the height of the cursor.                                                                                  |
| int           | **hot_x** | the x-axis offset from the left of the cursor image to the mouse x position, in the range of 0 to `w` - 1. |
| int           | **hot_y** | the y-axis offset from the top of the cursor image to the mouse y position, in the range of 0 to `h` - 1.  |

## Return Value

([SDL_Cursor](SDL_Cursor) *) Returns a new cursor with the specified
parameters on success or NULL on failure; call
[SDL_GetError](SDL_GetError)() for more information.

## Remarks

`mask` has to be in MSB (Most Significant Bit) format.

The cursor width (`w`) must be a multiple of 8 bits.

The cursor is created in black and white according to the following:

- data=0, mask=1: white
- data=1, mask=1: black
- data=0, mask=0: transparent
- data=1, mask=0: inverted color if possible, black if not.

Cursors created with this function must be freed with
[SDL_DestroyCursor](SDL_DestroyCursor)().

If you want to have a color cursor, or create your cursor from an
[SDL_Surface](SDL_Surface), you should use
[SDL_CreateColorCursor](SDL_CreateColorCursor)(). Alternately, you can hide
the cursor and draw your own as part of your game's rendering, but it will
be bound to the framerate.

Also, [SDL_CreateSystemCursor](SDL_CreateSystemCursor)() is available,
which provides several readily-available system cursors to pick from.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_CreateAnimatedCursor](SDL_CreateAnimatedCursor)
- [SDL_CreateColorCursor](SDL_CreateColorCursor)
- [SDL_CreateSystemCursor](SDL_CreateSystemCursor)
- [SDL_DestroyCursor](SDL_DestroyCursor)
- [SDL_SetCursor](SDL_SetCursor)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryMouse](CategoryMouse)

