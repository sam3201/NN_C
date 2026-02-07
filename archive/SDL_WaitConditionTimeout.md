# SDL_WaitConditionTimeout

Wait until a condition variable is signaled or a certain time has passed.

## Header File

Defined in [<SDL3/SDL_mutex.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_mutex.h)

## Syntax

```c
bool SDL_WaitConditionTimeout(SDL_Condition *cond,
                    SDL_Mutex *mutex, Sint32 timeoutMS);
```

## Function Parameters

|                                  |               |                                                                        |
| -------------------------------- | ------------- | ---------------------------------------------------------------------- |
| [SDL_Condition](SDL_Condition) * | **cond**      | the condition variable to wait on.                                     |
| [SDL_Mutex](SDL_Mutex) *         | **mutex**     | the mutex used to coordinate thread access.                            |
| [Sint32](Sint32)                 | **timeoutMS** | the maximum time to wait, in milliseconds, or -1 to wait indefinitely. |

## Return Value

(bool) Returns true if the condition variable is signaled, false if the
condition is not signaled in the allotted time.

## Remarks

This function unlocks the specified `mutex` and waits for another thread to
call [SDL_SignalCondition](SDL_SignalCondition)() or
[SDL_BroadcastCondition](SDL_BroadcastCondition)() on the condition
variable `cond`, or for the specified time to elapse. Once the condition
variable is signaled or the time elapsed, the mutex is re-locked and the
function returns.

The mutex must be locked before calling this function. Locking the mutex
recursively (more than once) is not supported and leads to undefined
behavior.

## Thread Safety

It is safe to call this function from any thread.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_BroadcastCondition](SDL_BroadcastCondition)
- [SDL_SignalCondition](SDL_SignalCondition)
- [SDL_WaitCondition](SDL_WaitCondition)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryMutex](CategoryMutex)

