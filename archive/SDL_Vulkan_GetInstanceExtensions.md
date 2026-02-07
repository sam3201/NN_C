# SDL_Vulkan_GetInstanceExtensions

Get the Vulkan instance extensions needed for vkCreateInstance.

## Header File

Defined in [<SDL3/SDL_vulkan.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_vulkan.h)

## Syntax

```c
char const * const * SDL_Vulkan_GetInstanceExtensions(Uint32 *count);
```

## Function Parameters

|                    |           |                                                             |
| ------------------ | --------- | ----------------------------------------------------------- |
| [Uint32](Uint32) * | **count** | a pointer filled in with the number of extensions returned. |

## Return Value

(char const * const *) Returns an array of extension name strings on
success, NULL on failure; call [SDL_GetError](SDL_GetError)() for more
information.

## Remarks

This should be called after either calling
[SDL_Vulkan_LoadLibrary](SDL_Vulkan_LoadLibrary)() or creating an
[SDL_Window](SDL_Window) with the [`SDL_WINDOW_VULKAN`](SDL_WINDOW_VULKAN)
flag.

On return, the variable pointed to by `count` will be set to the number of
elements returned, suitable for using with
VkInstanceCreateInfo::enabledExtensionCount, and the returned array can be
used with VkInstanceCreateInfo::ppEnabledExtensionNames, for calling
Vulkan's vkCreateInstance API.

You should not free the returned array; it is owned by SDL.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_Vulkan_CreateSurface](SDL_Vulkan_CreateSurface)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryVulkan](CategoryVulkan)

