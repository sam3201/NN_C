# SDL_CreateTexture

Create a texture for a rendering context.

## Header File

Defined in [<SDL3/SDL_render.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_render.h)

## Syntax

```c
SDL_Texture * SDL_CreateTexture(SDL_Renderer *renderer, SDL_PixelFormat format, SDL_TextureAccess access, int w, int h);
```

## Function Parameters

|                                        |              |                                                                         |
| -------------------------------------- | ------------ | ----------------------------------------------------------------------- |
| [SDL_Renderer](SDL_Renderer) *         | **renderer** | the rendering context.                                                  |
| [SDL_PixelFormat](SDL_PixelFormat)     | **format**   | one of the enumerated values in [SDL_PixelFormat](SDL_PixelFormat).     |
| [SDL_TextureAccess](SDL_TextureAccess) | **access**   | one of the enumerated values in [SDL_TextureAccess](SDL_TextureAccess). |
| int                                    | **w**        | the width of the texture in pixels.                                     |
| int                                    | **h**        | the height of the texture in pixels.                                    |

## Return Value

([SDL_Texture](SDL_Texture) *) Returns the created texture or NULL on
failure; call [SDL_GetError](SDL_GetError)() for more information.

## Remarks

The contents of a texture when first created are not defined.

## Thread Safety

This function should only be called on the main thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_CreateTextureFromSurface](SDL_CreateTextureFromSurface)
- [SDL_CreateTextureWithProperties](SDL_CreateTextureWithProperties)
- [SDL_DestroyTexture](SDL_DestroyTexture)
- [SDL_GetTextureSize](SDL_GetTextureSize)
- [SDL_UpdateTexture](SDL_UpdateTexture)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryRender](CategoryRender)

