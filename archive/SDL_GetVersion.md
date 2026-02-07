# SDL_GetVersion

Get the version of SDL that is linked against your program.

## Header File

Defined in [<SDL3/SDL_version.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_version.h)

## Syntax

```c
int SDL_GetVersion(void);
```

## Return Value

(int) Returns the version of the linked library.

## Remarks

If you are linking to SDL dynamically, then it is possible that the current
version will be different than the version you compiled against. This
function returns the current version, while [SDL_VERSION](SDL_VERSION) is
the version you compiled with.

This function may be called safely at any time, even before
[SDL_Init](SDL_Init)().

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_GetRevision](SDL_GetRevision)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryVersion](CategoryVersion)

