# work_graphs

This example draws a Koch snowflake and a Sierpinski carpet with a work graph. Lines and squares are split further the closer they are to a moving focus point, so the GPU decides how much work each part of the image needs.

The graph uses every node launch mode:

- `Entry` is a thread launch node that receives one record per fractal from the CPU.
- `Snowflake` is a recursive thread launch node that splits lines.
- `DrawLine` is a coalescing node that draws up to 32 lines per thread group.
- `Carpet` is a recursive broadcasting node with a fixed dispatch grid that splits squares.
- `FillSquare` is a broadcasting node whose dispatch grid depends on the size of the square.

On DX12, work graphs need shader model 6.8. The HLSL is compiled with DXC, which must support shader model 6.8, for example by pointing `WGPU_DX12_COMPILER` to a recent `dxcompiler.dll`.

On Vulkan, work graphs need `VK_AMDX_shader_enqueue`. `shader.spv` is the HLSL compiled with `dxc -spirv -T lib_6_8 -fspv-target-env=vulkan1.3 shader.hlsl -Fo shader.spv`.

## To Run

```
cargo run --bin wgpu-examples work_graphs
```

## Screenshots

![Work graphs example](./screenshot.png)
