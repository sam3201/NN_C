# SDL_CreateCondition

Create a condition variable.

## Header File

Defined in [<SDL3/SDL_mutex.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_mutex.h)

## Syntax

```c
SDL_Condition * SDL_CreateCondition(void);
```

## Return Value

([SDL_Condition](SDL_Condition) *) Returns a new condition variable or NULL
on failure; call [SDL_GetError](SDL_GetError)() for more information.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_BroadcastCondition](SDL_BroadcastCondition)
- [SDL_SignalCondition](SDL_SignalCondition)
- [SDL_WaitCondition](SDL_WaitCondition)
- [SDL_WaitConditionTimeout](SDL_WaitConditionTimeout)
- [SDL_DestroyCondition](SDL_DestroyCondition)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryMutex](CategoryMutex)

