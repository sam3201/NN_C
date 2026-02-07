# SDL_GetTicks

Get the number of milliseconds that have elapsed since the SDL library initialization.

## Header File

Defined in [<SDL3/SDL_timer.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_timer.h)

## Syntax

```c
Uint64 SDL_GetTicks(void);
```

## Return Value

([Uint64](Uint64)) Returns an unsigned 64‑bit integer that represents the
number of milliseconds that have elapsed since the SDL library was
initialized (typically via a call to [SDL_Init](SDL_Init)).

## Thread Safety

It is safe to call this function from any thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_GetTicksNS](SDL_GetTicksNS)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryTimer](CategoryTimer)

