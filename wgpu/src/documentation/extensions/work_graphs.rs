/*!
# 🧪Experimental🧪 Work Graphs

`wgpu` supports an experimental version of [D3D12 work graphs] when
[`Features::EXPERIMENTAL_WORK_GRAPHS`] is enabled.

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes.

## Overview

A work graph is a graph of compute shader nodes. Nodes send records to other nodes, and the
GPU schedules the work that the records create. The application starts a graph by sending
records to its entrypoint nodes.

Work graphs are only available on DX12 devices that support work graphs tier 1.0 and shader
model 6.8. Some systems need the D3D12 Agility SDK, which can be loaded with
[`Dx12BackendOptions::agility_sdk`].

## Shaders

`naga` does not support work graphs, so node shaders must be passed through with
[`Device::create_shader_module_passthrough`], which requires [`Features::PASSTHROUGH_SHADERS`].
The shader module contains a shader library in one of these forms:

- DXIL compiled for a `lib_6_8` or newer target. The module should list no entry points.
- HLSL, which `wgpu` compiles with DXC for the `lib` target of the device's shader model.
  It uses DXC's default HLSL version.

Resources must use the registers that `wgpu` assigns to the [`PipelineLayout`]. Read-write storage
buffers are raw UAVs, so they must be declared as `RWByteAddressBuffer`.

Local root arguments are not supported.

## `wgpu` API

[`Device::create_work_graph_pipeline`] creates a [`WorkGraphPipeline`] from a shader library and
the name of the work graph program. The pipeline reports the size of the backing memory it needs
with [`WorkGraphPipeline::memory_requirements`] and its entrypoints with
[`WorkGraphPipeline::entrypoints`].

[`ComputePass::set_work_graph_pipeline`] sets the work graph and the buffer region that holds its
backing memory. The buffer needs [`BufferUsages::STORAGE`]. The backing memory cannot be used by
any other binding in the same dispatch. `wgpu` initializes the backing memory as needed, so it
can be shared by different work graphs.

[`ComputePass::dispatch_graph`] sends records to one or more entrypoints. Records use the C
struct layout rules, and `wgpu` checks their size, stride and alignment against the entrypoint.
`dispatch_graph` is `unsafe`, because `wgpu` cannot check the contents of the records against
the limits declared by the nodes. For example, an `SV_DispatchGrid` value larger than the node's
`NodeMaxDispatchGrid` is undefined behavior.

Bind groups and immediates are shared with compute pipelines, because work graphs use the
compute root signature.

```rust,ignore
let pipeline = device.create_work_graph_pipeline(&wgpu::WorkGraphPipelineDescriptor {
    label: Some("work graph"),
    layout: &pipeline_layout,
    module: &library,
    program_name: "graph",
    entrypoints: &[],
    include_all_available_nodes: true,
});

let backing_memory = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("work graph backing memory"),
    size: pipeline.memory_requirements().min_size,
    usage: wgpu::BufferUsages::STORAGE,
    mapped_at_creation: false,
});

let mut pass = encoder.begin_compute_pass(&Default::default());
pass.set_bind_group(0, &bind_group, &[]);
pass.set_work_graph_pipeline(&pipeline, backing_memory.slice(..));
unsafe {
    pass.dispatch_graph(&[wgpu::WorkGraphNodeInput {
        entrypoint_index: 0,
        num_records: records.len() as u32,
        record_stride: size_of::<Record>() as u64,
        records: bytemuck::cast_slice(&records),
    }]);
}
```

## Not yet supported

- Records generated on the GPU (`D3D12_DISPATCH_MODE_NODE_GPU_INPUT`).
- Overriding node properties in the work graph description.
- Mesh nodes.

[D3D12 work graphs]: https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs.html
*/

use crate::{
    BufferUsages, ComputePass, Device, Dx12BackendOptions, Features, PipelineLayout,
    WorkGraphPipeline,
};
