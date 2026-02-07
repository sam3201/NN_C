# SDL_CreateThread

Create a new thread with a default stack size.

## Header File

Defined in [<SDL3/SDL_thread.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_thread.h)

## Syntax

```c
SDL_Thread * SDL_CreateThread(SDL_ThreadFunction fn, const char *name, void *data);
```

## Function Parameters

|                                          |          |                                                                                  |
| ---------------------------------------- | -------- | -------------------------------------------------------------------------------- |
| [SDL_ThreadFunction](SDL_ThreadFunction) | **fn**   | the [SDL_ThreadFunction](SDL_ThreadFunction) function to call in the new thread. |
| const char *                             | **name** | the name of the thread.                                                          |
| void *                                   | **data** | a pointer that is passed to `fn`.                                                |

## Return Value

([SDL_Thread](SDL_Thread) *) Returns an opaque pointer to the new thread
object on success, NULL if the new thread could not be created; call
[SDL_GetError](SDL_GetError)() for more information.

## Remarks

This is a convenience function, equivalent to calling
[SDL_CreateThreadWithProperties](SDL_CreateThreadWithProperties) with the
following properties set:

- [`SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER`](SDL_PROP_THREAD_CREATE_ENTRY_FUNCTION_POINTER):
  `fn`
- [`SDL_PROP_THREAD_CREATE_NAME_STRING`](SDL_PROP_THREAD_CREATE_NAME_STRING):
  `name`
- [`SDL_PROP_THREAD_CREATE_USERDATA_POINTER`](SDL_PROP_THREAD_CREATE_USERDATA_POINTER):
  `data`

Note that this "function" is actually a macro that calls an internal
function with two extra parameters not listed here; they are hidden through
preprocessor macros and are needed to support various C runtimes at the
point of the function call. Language bindings that aren't using the C
headers will need to deal with this.

Usually, apps should just call this function the same way on every platform
and let the macros hide the details.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_CreateThreadWithProperties](SDL_CreateThreadWithProperties)
- [SDL_WaitThread](SDL_WaitThread)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryThread](CategoryThread)

