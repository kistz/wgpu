/*!
# 🧪Experimental🧪 Work Graphs

`wgpu` supports an experimental version of [D3D12 work graphs] and of the Vulkan extension
[`VK_AMDX_shader_enqueue`] when [`Features::EXPERIMENTAL_WORK_GRAPHS`] is enabled.

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes.

## Overview

A work graph is a graph of compute shader nodes. Nodes send records to other nodes, and the
GPU schedules the work that the records create. The application starts a graph by sending
records to its entrypoint nodes.

On DX12, work graphs are available on devices that support work graphs tier 1.0 and shader
model 6.8. Some systems need the D3D12 Agility SDK, which can be loaded with
[`Dx12BackendOptions::agility_sdk`].

On Vulkan, work graphs are available on Vulkan 1.3 devices that support revision 2 of the
provisional `VK_AMDX_shader_enqueue` extension, `VK_KHR_maintenance5`, `VK_KHR_pipeline_library`
and `VK_KHR_buffer_device_address`.

## Shaders

`naga` does not support work graphs, so node shaders must be passed through with
[`Device::create_shader_module_passthrough`], which requires [`Features::PASSTHROUGH_SHADERS`].
On DX12, the shader module contains a shader library in one of these forms:

- DXIL compiled for a `lib_6_8` or newer target. The module should list no entry points.
- HLSL, which `wgpu` compiles with DXC for the `lib` target of the device's shader model.
  It uses DXC's default HLSL version.

Resources must use the registers that `wgpu` assigns to the [`PipelineLayout`]. Read-write storage
buffers are raw UAVs, so they must be declared as `RWByteAddressBuffer`.

Local root arguments are not supported.

On Vulkan, the shader module contains SPIR-V that uses `SPV_AMDX_shader_enqueue`. DXC compiles
HLSL node shaders to it with `dxc -spirv -T lib_6_8 -fspv-target-env=vulkan1.3`. Resources must use
the group and binding of the [`PipelineLayout`], for example with `[[vk::binding(binding, group)]]`.
Vulkan has no query for the input records of a node, so `wgpu` reads the nodes from the SPIR-V,
which differs from DX12 in some ways:

- The name of a node is the name of its entry point, because DXC does not keep the name from
  `[NodeID]` in SPIR-V. The array index from `[NodeID]` is kept. Nodes that send records to a
  name that no entry point has are an error, so node arrays are not supported yet.
- DXC does not keep `[NodeIsProgramEntry]`. A node that other nodes send records to only becomes
  an entrypoint if it is listed in [`WorkGraphPipelineDescriptor::entrypoints`].
- [`WorkGraphPipelineDescriptor::program_name`] is ignored.

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

### Input in GPU memory

[`ComputePass::dispatch_graph_indirect`] reads the input from a buffer when the dispatch executes,
so earlier GPU commands can generate it. The buffer holds a [`WorkGraphGpuNodeInput`] for one
entrypoint, or a [`WorkGraphGpuMultiNodeInput`] that points to an array of them. A node input
refers to its entrypoint by [`WorkGraphEntrypoint::gpu_input_index`], which is the entrypoint
index on DX12 and the node index that the driver assigns on Vulkan.

On Vulkan, the number of records of a node input is limited by the device's
`maxExecutionGraphShaderPayloadCount`, which `wgpu` does not report yet. `wgpu` splits input in
CPU memory to stay within it.

This input refers to records and node inputs by GPU address. [`Buffer::work_graph_input_address`]
returns the address of a buffer, which does not change while the buffer exists. Every buffer that
the input refers to must be passed to `dispatch_graph_indirect`, so that `wgpu` keeps it alive and
inserts barriers between it and other commands. `wgpu` cannot check what the GPU reads, so the
caller must make sure that the input is valid when the dispatch executes.

Traces that are replayed by the `player` contain the addresses from the original run, so GPU input
is not replayed correctly.

```rust,ignore
let pipeline = device.create_work_graph_pipeline(&wgpu::WorkGraphPipelineDescriptor {
    label: Some("work graph"),
    layout: &pipeline_layout,
    module: &library,
    program_name: "graph",
    entrypoints: &[],
    include_all_available_nodes: true,
    mesh_nodes: &[],
    max_input_records: None,
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

let input = wgpu::WorkGraphGpuNodeInput {
    gpu_input_index: pipeline.entrypoints()[0].gpu_input_index,
    num_records: records.len() as u32,
    records_address: records_buffer.work_graph_input_address(),
    record_stride: size_of::<Record>() as u64,
};
let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    label: Some("work graph input"),
    contents: input.as_bytes(),
    usage: wgpu::BufferUsages::STORAGE,
});
unsafe {
    pass.dispatch_graph_indirect(
        &input_buffer,
        0,
        wgpu::WorkGraphGpuInputMode::SingleNode,
        &[&records_buffer],
    );
}
```

## Mesh nodes

With [`Features::EXPERIMENTAL_WORK_GRAPHS_1_1`], a work graph can contain mesh nodes, which are
leaf nodes that draw like mesh shaders. They are only supported on DX12. D3D12 only supports them
in preview Agility SDKs, on drivers that report work graphs tier 1.1. Node shaders with `[NodeLaunch("mesh")]` need a DXC
version with mesh node support, so precompiled DXIL may be the easier option.

Each [`WorkGraphMeshNode`] names a mesh node shader and an optional fragment shader in the
shader library, and has the same primitive, depth-stencil and multisample state as a render
pipeline. The node's ID in the work graph is [`WorkGraphMeshNode::node`], and other nodes send
records to it by that ID. All mesh nodes of a work graph must render to the same attachment
formats with the same sample count.

A work graph with mesh nodes runs in a [`RenderPass`] whose attachments match the mesh nodes,
with [`RenderPass::set_work_graph_pipeline`] and [`RenderPass::dispatch_graph`]. All nodes use the
bind groups and immediates of the render pass, so the visibility of a bind group layout must
include every stage that uses it: `COMPUTE` for compute nodes, `MESH` for mesh nodes and
`FRAGMENT` for fragment shaders. Setting the work graph replaces the render pipeline, so a render
pipeline must be set again before the next draw.

D3D12 needs to know the size of the largest dispatch to compute the backing memory size, so
[`WorkGraphPipelineDescriptor::max_input_records`] is required with mesh nodes, and every
[`RenderPass::dispatch_graph`] call must stay within it.

## Not yet supported

- Overriding node properties in the work graph description.
- Input in GPU memory for work graphs with mesh nodes.
- Work graphs in render bundles.
- Mesh nodes and node arrays on Vulkan.

[D3D12 work graphs]: https://microsoft.github.io/DirectX-Specs/d3d/WorkGraphs.html
[`VK_AMDX_shader_enqueue`]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMDX_shader_enqueue.html
*/

use crate::{
    Buffer, BufferUsages, ComputePass, Device, Dx12BackendOptions, Features, PipelineLayout,
    RenderPass, WorkGraphEntrypoint, WorkGraphGpuMultiNodeInput, WorkGraphGpuNodeInput,
    WorkGraphMeshNode, WorkGraphPipeline, WorkGraphPipelineDescriptor,
};
