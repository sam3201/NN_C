# SDL_BeginGPURenderPass

Begins a render pass on a command buffer.

## Header File

Defined in [<SDL3/SDL_gpu.h>](https://github.com/libsdl-org/SDL/blob/main/include/SDL3/SDL_gpu.h)

## Syntax

```c
SDL_GPURenderPass * SDL_BeginGPURenderPass(
    SDL_GPUCommandBuffer *command_buffer,
    const SDL_GPUColorTargetInfo *color_target_infos,
    Uint32 num_color_targets,
    const SDL_GPUDepthStencilTargetInfo *depth_stencil_target_info);
```

## Function Parameters

|                                                                        |                               |                                                                                       |
| ---------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------- |
| [SDL_GPUCommandBuffer](SDL_GPUCommandBuffer) *                         | **command_buffer**            | a command buffer.                                                                     |
| const [SDL_GPUColorTargetInfo](SDL_GPUColorTargetInfo) *               | **color_target_infos**        | an array of texture subresources with corresponding clear values and load/store ops.  |
| [Uint32](Uint32)                                                       | **num_color_targets**         | the number of color targets in the color_target_infos array.                          |
| const [SDL_GPUDepthStencilTargetInfo](SDL_GPUDepthStencilTargetInfo) * | **depth_stencil_target_info** | a texture subresource with corresponding clear value and load/store ops, may be NULL. |

## Return Value

([SDL_GPURenderPass](SDL_GPURenderPass) *) Returns a render pass handle.

## Remarks

A render pass consists of a set of texture subresources (or depth slices in
the 3D texture case) which will be rendered to during the render pass,
along with corresponding clear values and load/store operations. All
operations related to graphics pipelines must take place inside of a render
pass. A default viewport and scissor state are automatically set when this
is called. You cannot begin another render pass, or begin a compute pass or
copy pass until you have ended the render pass.

Using [SDL_GPU_LOADOP_LOAD](SDL_GPU_LOADOP_LOAD) before any contents have
been written to the texture subresource will result in undefined behavior.
[SDL_GPU_LOADOP_CLEAR](SDL_GPU_LOADOP_CLEAR) will set the contents of the
texture subresource to a single value before any rendering is performed.
It's fine to do an empty render pass using
[SDL_GPU_STOREOP_STORE](SDL_GPU_STOREOP_STORE) to clear a texture, but in
general it's better to think of clearing not as an independent operation
but as something that's done as the beginning of a render pass.

## Version

This function is available since SDL 3.2.0.

## See Also

- [SDL_EndGPURenderPass](SDL_EndGPURenderPass)

----
[CategoryAPI](CategoryAPI), [CategoryAPIFunction](CategoryAPIFunction), [CategoryGPU](CategoryGPU)

