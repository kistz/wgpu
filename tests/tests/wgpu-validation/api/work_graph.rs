//! Tests of [`wgpu::WorkGraphPipeline`] validation.
//!
//! The noop backend reports a minimum backing memory size of 256 bytes, and one
//! entrypoint for each node listed in [`WorkGraphPipelineDescriptor::entrypoints`],
//! each with a record size of 16 bytes and a record alignment of 4 bytes.
//! Mesh node tests use the same pipelines with an `Rgba8Unorm` mesh node.

use std::borrow::Cow;

use wgpu::*;
use wgpu_test::{fail, valid};

const MIN_BACKING_MEMORY_SIZE: u64 = 256;

fn device(features: Features) -> (Device, Queue) {
    Device::noop(&DeviceDescriptor {
        required_features: features,
        experimental_features: unsafe { ExperimentalFeatures::enabled() },
        ..Default::default()
    })
}

fn work_graph_device() -> (Device, Queue) {
    device(Features::EXPERIMENTAL_WORK_GRAPHS | Features::PASSTHROUGH_SHADERS)
}

fn library(device: &Device) -> ShaderModule {
    unsafe {
        device.create_shader_module_passthrough(ShaderModuleDescriptorPassthrough {
            label: Some("work graph library"),
            dxil: Some(Cow::Borrowed(&[0; 4])),
            ..Default::default()
        })
    }
}

fn layout(device: &Device, bind_group_layout: Option<&BindGroupLayout>) -> PipelineLayout {
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    })
}

fn pipeline_with_layout(device: &Device, layout: &PipelineLayout) -> WorkGraphPipeline {
    let module = library(device);
    device.create_work_graph_pipeline(&WorkGraphPipelineDescriptor {
        label: Some("work graph"),
        layout,
        module: &module,
        program_name: "graph",
        entrypoints: &[WorkGraphNodeId {
            name: Cow::Borrowed("entry"),
            array_index: 0,
        }],
        include_all_available_nodes: true,
        mesh_nodes: &[],
        max_input_records: None,
    })
}

fn pipeline(device: &Device) -> WorkGraphPipeline {
    pipeline_with_layout(device, &layout(device, None))
}

fn backing_memory(device: &Device, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("backing memory"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn input(
    entrypoint_index: u32,
    num_records: u32,
    record_stride: u64,
    records: &[u8],
) -> WorkGraphNodeInput<&[u8]> {
    WorkGraphNodeInput {
        entrypoint_index,
        num_records,
        record_stride,
        records,
    }
}

fn storage_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Encodes a pass that sets `pipeline` with `backing_memory` and dispatches `inputs`.
fn encode_dispatch(
    device: &Device,
    pipeline: &WorkGraphPipeline,
    backing_memory: BufferSlice<'_>,
    inputs: &[WorkGraphNodeInput<&[u8]>],
) -> CommandBuffer {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_work_graph_pipeline(pipeline, backing_memory);
        unsafe { pass.dispatch_graph(inputs) };
    }
    encoder.finish()
}

#[test]
fn create_requires_work_graphs_feature() {
    let (device, _queue) = device(Features::PASSTHROUGH_SHADERS);
    fail(
        &device,
        || pipeline(&device),
        Some("EXPERIMENTAL_WORK_GRAPHS"),
    );
}

#[test]
fn create_requires_passthrough_module() {
    let (device, _queue) = work_graph_device();
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });
    let layout = layout(&device, None);
    fail(
        &device,
        || {
            device.create_work_graph_pipeline(&WorkGraphPipelineDescriptor {
                label: None,
                layout: &layout,
                module: &module,
                program_name: "graph",
                entrypoints: &[],
                include_all_available_nodes: true,
                mesh_nodes: &[],
                max_input_records: None,
            })
        },
        Some("passthrough shader module"),
    );
}

#[test]
fn create_reports_entrypoints_and_memory_requirements() {
    let (device, _queue) = work_graph_device();
    let pipeline = valid(&device, || pipeline(&device));

    assert_eq!(
        pipeline.memory_requirements().min_size,
        MIN_BACKING_MEMORY_SIZE
    );
    let entry = WorkGraphNodeId {
        name: Cow::Borrowed("entry"),
        array_index: 0,
    };
    assert_eq!(pipeline.entrypoint_index(&entry), Some(0));
    assert_eq!(
        pipeline.entrypoint_index(&WorkGraphNodeId {
            name: Cow::Borrowed("other"),
            array_index: 0,
        }),
        None
    );
}

#[test]
fn dispatch_valid() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let records = [0u8; 32];

    valid(&device, || {
        encode_dispatch(
            &device,
            &pipeline,
            buffer.slice(..),
            &[input(0, 2, 16, &records)],
        )
    });
    // A stride of zero repeats the first record.
    valid(&device, || {
        encode_dispatch(
            &device,
            &pipeline,
            buffer.slice(..),
            &[input(0, 5, 0, &records[..16])],
        )
    });
}

#[test]
fn dispatch_without_work_graph_pipeline() {
    let (device, _queue) = work_graph_device();
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                unsafe { pass.dispatch_graph(&[input(0, 1, 0, &[0; 16])]) };
            }
            encoder.finish()
        },
        Some("Work graph pipeline must be set"),
    );
}

#[test]
fn backing_memory_requires_storage_usage() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::COPY_DST);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..),
                &[input(0, 1, 0, &[0; 16])],
            )
        },
        Some("STORAGE"),
    );
}

#[test]
fn backing_memory_too_small() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..MIN_BACKING_MEMORY_SIZE - 8),
                &[input(0, 1, 0, &[0; 16])],
            )
        },
        Some("smaller than the minimum size"),
    );
}

#[test]
fn backing_memory_unaligned_offset() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, 2 * MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(4..),
                &[input(0, 1, 0, &[0; 16])],
            )
        },
        Some("is not a multiple of 8"),
    );
}

#[test]
fn backing_memory_bound_as_storage() {
    let (device, _queue) = work_graph_device();
    let bind_group_layout = storage_bind_group_layout(&device);
    let pipeline = pipeline_with_layout(&device, &layout(&device, Some(&bind_group_layout)));
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_work_graph_pipeline(&pipeline, buffer.slice(..));
                pass.set_bind_group(0, &bind_group, &[]);
                unsafe { pass.dispatch_graph(&[input(0, 1, 0, &[0; 16])]) };
            }
            encoder.finish()
        },
        Some("WORK_GRAPH_BACKING_MEMORY"),
    );
}

#[test]
fn dispatch_invalid_entrypoint() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..),
                &[input(1, 1, 0, &[0; 16])],
            )
        },
        Some("Entrypoint index 1 of input 0 is out of range"),
    );
}

#[test]
fn dispatch_record_stride_too_small() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..),
                &[input(0, 2, 8, &[0; 32])],
            )
        },
        Some("smaller than the record size"),
    );
}

#[test]
fn dispatch_record_stride_unaligned() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..),
                &[input(0, 2, 18, &[0; 64])],
            )
        },
        Some("is not a multiple of 4"),
    );
}

#[test]
fn dispatch_records_too_small() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                buffer.slice(..),
                &[input(0, 2, 16, &[0; 31])],
            )
        },
        Some("provides 31 bytes of records, but 32 bytes are required"),
    );
}

#[test]
fn dispatch_workgroups_with_work_graph_pipeline() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_work_graph_pipeline(&pipeline, buffer.slice(..));
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.finish()
        },
        Some("Compute pipeline must be set"),
    );
}

#[test]
fn compute_pipeline_replaces_work_graph_pipeline() {
    let (device, _queue) = work_graph_device();
    let work_graph = pipeline(&device);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });
    let compute = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);

    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_work_graph_pipeline(&work_graph, buffer.slice(..));
                pass.set_pipeline(&compute);
                unsafe { pass.dispatch_graph(&[input(0, 1, 0, &[0; 16])]) };
            }
            encoder.finish()
        },
        Some("Work graph pipeline must be set"),
    );

    // Setting the same compute pipeline again after a work graph pipeline must not be skipped.
    valid(&device, || {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&compute);
            pass.set_work_graph_pipeline(&work_graph, buffer.slice(..));
            pass.set_pipeline(&compute);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.finish()
    });
}

fn gpu_input_buffer(device: &Device, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("GPU input"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Encodes a pass that sets `pipeline` with `backing_memory` and dispatches the GPU input that
/// `input` holds at `offset`.
fn encode_dispatch_indirect(
    device: &Device,
    pipeline: &WorkGraphPipeline,
    backing_memory: BufferSlice<'_>,
    input: &Buffer,
    offset: BufferAddress,
    mode: WorkGraphGpuInputMode,
    referenced_buffers: &[&Buffer],
) -> CommandBuffer {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_work_graph_pipeline(pipeline, backing_memory);
        unsafe { pass.dispatch_graph_indirect(input, offset, mode, referenced_buffers) };
    }
    encoder.finish()
}

#[test]
fn gpu_input_layout() {
    assert_eq!(WorkGraphGpuInputMode::SingleNode.input_size(), 24);
    assert_eq!(WorkGraphGpuInputMode::MultiNode.input_size(), 24);

    let input = WorkGraphGpuMultiNodeInput::new(2, 0x1000, 24);
    assert_eq!(input.as_bytes()[..4], 2u32.to_le_bytes());
    assert_eq!(input.as_bytes()[8..16], 0x1000u64.to_le_bytes());
}

#[test]
fn dispatch_indirect_valid() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 48, BufferUsages::STORAGE);
    let records = gpu_input_buffer(&device, 64, BufferUsages::STORAGE);

    valid(&device, || {
        encode_dispatch_indirect(
            &device,
            &pipeline,
            memory.slice(..),
            &input,
            0,
            WorkGraphGpuInputMode::SingleNode,
            &[&records],
        )
    });
    valid(&device, || {
        encode_dispatch_indirect(
            &device,
            &pipeline,
            memory.slice(..),
            &input,
            24,
            WorkGraphGpuInputMode::MultiNode,
            &[&input, &records],
        )
    });
}

#[test]
fn dispatch_indirect_without_work_graph_pipeline() {
    let (device, _queue) = work_graph_device();
    let input = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                unsafe {
                    pass.dispatch_graph_indirect(&input, 0, WorkGraphGpuInputMode::SingleNode, &[])
                };
            }
            encoder.finish()
        },
        Some("Work graph pipeline must be set"),
    );
}

#[test]
fn dispatch_indirect_requires_storage_usage() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let storage = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    let indirect = gpu_input_buffer(&device, 24, BufferUsages::INDIRECT);

    fail(
        &device,
        || {
            encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..),
                &indirect,
                0,
                WorkGraphGpuInputMode::SingleNode,
                &[],
            )
        },
        Some("STORAGE"),
    );
    fail(
        &device,
        || {
            encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..),
                &storage,
                0,
                WorkGraphGpuInputMode::SingleNode,
                &[&indirect],
            )
        },
        Some("STORAGE"),
    );
}

#[test]
fn dispatch_indirect_unaligned_offset() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 48, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..),
                &input,
                4,
                WorkGraphGpuInputMode::SingleNode,
                &[],
            )
        },
        Some("GPU input offset 4 is not a multiple of 8"),
    );
}

#[test]
fn dispatch_indirect_overrun() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 32, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..),
                &input,
                16,
                WorkGraphGpuInputMode::MultiNode,
                &[],
            )
        },
        Some("would overrun buffer of size 32"),
    );
}

#[test]
fn dispatch_indirect_destroyed_referenced_buffer() {
    let (device, queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    let records = gpu_input_buffer(&device, 64, BufferUsages::STORAGE);
    records.destroy();
    fail(
        &device,
        || {
            queue.submit([encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..),
                &input,
                0,
                WorkGraphGpuInputMode::SingleNode,
                &[&records],
            )])
        },
        Some("has been destroyed"),
    );
}

#[test]
fn dispatch_indirect_references_backing_memory() {
    let (device, _queue) = work_graph_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, 2 * MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    fail(
        &device,
        || {
            encode_dispatch_indirect(
                &device,
                &pipeline,
                memory.slice(..MIN_BACKING_MEMORY_SIZE),
                &input,
                0,
                WorkGraphGpuInputMode::SingleNode,
                &[&memory],
            )
        },
        Some("WORK_GRAPH_BACKING_MEMORY"),
    );
}

#[test]
fn dispatch_indirect_input_bound_as_storage() {
    let (device, _queue) = work_graph_device();
    let bind_group_layout = storage_bind_group_layout(&device);
    let pipeline = pipeline_with_layout(&device, &layout(&device, Some(&bind_group_layout)));
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let input = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: input.as_entire_binding(),
        }],
    });

    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_work_graph_pipeline(&pipeline, memory.slice(..));
                pass.set_bind_group(0, &bind_group, &[]);
                unsafe {
                    pass.dispatch_graph_indirect(&input, 0, WorkGraphGpuInputMode::SingleNode, &[])
                };
            }
            encoder.finish()
        },
        Some("STORAGE_READ_ONLY"),
    );
}

#[test]
fn work_graph_input_address() {
    let (device, _queue) = work_graph_device();
    let buffer = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    valid(&device, || buffer.work_graph_input_address());
}

#[test]
#[should_panic(expected = "EXPERIMENTAL_WORK_GRAPHS")]
fn work_graph_input_address_requires_feature() {
    let (device, _queue) = device(Features::empty());
    let buffer = gpu_input_buffer(&device, 24, BufferUsages::STORAGE);
    buffer.work_graph_input_address();
}

#[test]
#[should_panic(expected = "STORAGE")]
fn work_graph_input_address_requires_storage_usage() {
    let (device, _queue) = work_graph_device();
    let buffer = gpu_input_buffer(&device, 24, BufferUsages::INDIRECT);
    buffer.work_graph_input_address();
}

fn mesh_node_device() -> (Device, Queue) {
    device(
        Features::EXPERIMENTAL_WORK_GRAPHS
            | Features::EXPERIMENTAL_WORK_GRAPHS_1_1
            | Features::PASSTHROUGH_SHADERS,
    )
}

const MAX_INPUT_RECORDS: WorkGraphMaxInputRecords = WorkGraphMaxInputRecords {
    records: 4,
    node_inputs: 1,
};

const RGBA8_TARGETS: &[Option<ColorTargetState>] = &[Some(ColorTargetState {
    format: TextureFormat::Rgba8Unorm,
    blend: None,
    write_mask: ColorWrites::ALL,
})];

fn mesh_node(targets: &[Option<ColorTargetState>]) -> WorkGraphMeshNode<'_> {
    WorkGraphMeshNode {
        node: WorkGraphNodeId {
            name: Cow::Borrowed("mesh"),
            array_index: 0,
        },
        mesh_entry_point: "mesh_main",
        fragment: Some(WorkGraphFragmentState {
            entry_point: "fragment_main",
            targets,
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    }
}

fn mesh_pipeline_with(
    device: &Device,
    layout: &PipelineLayout,
    mesh_nodes: &[WorkGraphMeshNode<'_>],
    max_input_records: Option<WorkGraphMaxInputRecords>,
) -> WorkGraphPipeline {
    let module = library(device);
    device.create_work_graph_pipeline(&WorkGraphPipelineDescriptor {
        label: Some("mesh work graph"),
        layout,
        module: &module,
        program_name: "graph",
        entrypoints: &[WorkGraphNodeId {
            name: Cow::Borrowed("entry"),
            array_index: 0,
        }],
        include_all_available_nodes: true,
        mesh_nodes,
        max_input_records,
    })
}

fn mesh_pipeline(device: &Device) -> WorkGraphPipeline {
    mesh_pipeline_with(
        device,
        &layout(device, None),
        &[mesh_node(RGBA8_TARGETS)],
        Some(MAX_INPUT_RECORDS),
    )
}

fn render_target(device: &Device, format: TextureFormat) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some("render target"),
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

fn begin_render_pass<'a>(encoder: &'a mut CommandEncoder, target: &TextureView) -> RenderPass<'a> {
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Encodes a render pass that draws to `target`, sets `pipeline` with `backing_memory` and
/// dispatches `inputs`.
fn encode_render_dispatch(
    device: &Device,
    target: &TextureView,
    pipeline: &WorkGraphPipeline,
    backing_memory: BufferSlice<'_>,
    inputs: &[WorkGraphNodeInput<&[u8]>],
) -> CommandBuffer {
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = begin_render_pass(&mut encoder, target);
        pass.set_work_graph_pipeline(pipeline, backing_memory);
        unsafe { pass.dispatch_graph(inputs) };
    }
    encoder.finish()
}

#[test]
fn mesh_nodes_require_feature() {
    let (device, _queue) = work_graph_device();
    fail(
        &device,
        || {
            mesh_pipeline_with(
                &device,
                &layout(&device, None),
                &[mesh_node(RGBA8_TARGETS)],
                Some(MAX_INPUT_RECORDS),
            )
        },
        Some("EXPERIMENTAL_WORK_GRAPHS_1_1"),
    );
}

#[test]
fn mesh_nodes_require_max_input_records() {
    let (device, _queue) = mesh_node_device();
    fail(
        &device,
        || {
            mesh_pipeline_with(
                &device,
                &layout(&device, None),
                &[mesh_node(RGBA8_TARGETS)],
                None,
            )
        },
        Some("max_input_records"),
    );
}

#[test]
fn mesh_node_requires_list_topology() {
    let (device, _queue) = mesh_node_device();
    let mut node = mesh_node(RGBA8_TARGETS);
    node.primitive.topology = PrimitiveTopology::TriangleStrip;
    fail(
        &device,
        || {
            mesh_pipeline_with(
                &device,
                &layout(&device, None),
                &[node],
                Some(MAX_INPUT_RECORDS),
            )
        },
        Some("list topology"),
    );
}

#[test]
fn mesh_node_validates_color_targets() {
    let (device, _queue) = mesh_node_device();
    let targets = [Some(ColorTargetState {
        format: TextureFormat::Depth32Float,
        blend: None,
        write_mask: ColorWrites::ALL,
    })];
    fail(
        &device,
        || {
            mesh_pipeline_with(
                &device,
                &layout(&device, None),
                &[mesh_node(&targets)],
                Some(MAX_INPUT_RECORDS),
            )
        },
        Some("Mesh node 0 is invalid"),
    );
}

#[test]
fn mesh_nodes_share_targets() {
    let (device, _queue) = mesh_node_device();
    let bgra8_targets = [Some(ColorTargetState::from(TextureFormat::Bgra8Unorm))];
    let mut second = mesh_node(&bgra8_targets);
    second.node.name = Cow::Borrowed("second");
    fail(
        &device,
        || {
            mesh_pipeline_with(
                &device,
                &layout(&device, None),
                &[mesh_node(RGBA8_TARGETS), second],
                Some(MAX_INPUT_RECORDS),
            )
        },
        Some("Mesh node 1 renders to different attachment formats"),
    );
}

#[test]
fn mesh_dispatch_valid() {
    let (device, _queue) = mesh_node_device();
    let pipeline = mesh_pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    let records = [0; 64];
    valid(&device, || {
        encode_render_dispatch(
            &device,
            &target,
            &pipeline,
            memory.slice(..),
            &[input(0, 4, 16, &records)],
        )
    });
}

#[test]
fn mesh_pipeline_in_compute_pass() {
    let (device, _queue) = mesh_node_device();
    let pipeline = mesh_pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let records = [0; 16];
    fail(
        &device,
        || {
            encode_dispatch(
                &device,
                &pipeline,
                memory.slice(..),
                &[input(0, 1, 16, &records)],
            )
        },
        Some("can only be used in a render pass"),
    );
}

#[test]
fn render_pass_requires_mesh_nodes() {
    let (device, _queue) = mesh_node_device();
    let pipeline = pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    let records = [0; 16];
    fail(
        &device,
        || {
            encode_render_dispatch(
                &device,
                &target,
                &pipeline,
                memory.slice(..),
                &[input(0, 1, 16, &records)],
            )
        },
        Some("without mesh nodes"),
    );
}

#[test]
fn mesh_dispatch_incompatible_targets() {
    let (device, _queue) = mesh_node_device();
    let pipeline = mesh_pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Bgra8Unorm);
    let records = [0; 16];
    fail(
        &device,
        || {
            encode_render_dispatch(
                &device,
                &target,
                &pipeline,
                memory.slice(..),
                &[input(0, 1, 16, &records)],
            )
        },
        Some("Incompatible color attachments"),
    );
}

#[test]
fn mesh_dispatch_depth_write_with_read_only_depth() {
    let (device, _queue) = mesh_node_device();
    let mut node = mesh_node(RGBA8_TARGETS);
    node.depth_stencil = Some(DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(CompareFunction::Less),
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    });
    let pipeline = mesh_pipeline_with(
        &device,
        &layout(&device, None),
        &[node],
        Some(MAX_INPUT_RECORDS),
    );
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    let depth = render_target(&device, TextureFormat::Depth32Float);
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: None,
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_work_graph_pipeline(&pipeline, memory.slice(..));
            }
            encoder.finish()
        },
        Some("read-only depth access"),
    );
}

#[test]
fn mesh_dispatch_too_many_records() {
    let (device, _queue) = mesh_node_device();
    let pipeline = mesh_pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    let records = [0; 16];
    fail(
        &device,
        || {
            encode_render_dispatch(
                &device,
                &target,
                &pipeline,
                memory.slice(..),
                &[input(0, 5, 0, &records)],
            )
        },
        Some("allows at most 4 records in 1 inputs"),
    );
    fail(
        &device,
        || {
            encode_render_dispatch(
                &device,
                &target,
                &pipeline,
                memory.slice(..),
                &[input(0, 1, 16, &records), input(0, 1, 16, &records)],
            )
        },
        Some("2 records in 2 inputs"),
    );
}

#[test]
fn draw_after_work_graph_requires_render_pipeline() {
    let (device, _queue) = mesh_node_device();
    let pipeline = mesh_pipeline(&device);
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = begin_render_pass(&mut encoder, &target);
                pass.set_work_graph_pipeline(&pipeline, memory.slice(..));
                pass.draw(0..3, 0..1);
            }
            encoder.finish()
        },
        Some("Render pipeline must be set"),
    );
}

#[test]
fn mesh_backing_memory_bound_as_storage() {
    let (device, _queue) = mesh_node_device();
    let bind_group_layout = storage_bind_group_layout(&device);
    let pipeline = mesh_pipeline_with(
        &device,
        &layout(&device, Some(&bind_group_layout)),
        &[mesh_node(RGBA8_TARGETS)],
        Some(MAX_INPUT_RECORDS),
    );
    let memory = backing_memory(&device, MIN_BACKING_MEMORY_SIZE, BufferUsages::STORAGE);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: memory.as_entire_binding(),
        }],
    });
    let target = render_target(&device, TextureFormat::Rgba8Unorm);
    let records = [0; 16];
    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut pass = begin_render_pass(&mut encoder, &target);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_work_graph_pipeline(&pipeline, memory.slice(..));
                unsafe { pass.dispatch_graph(&[input(0, 1, 16, &records)]) };
            }
            encoder.finish()
        },
        Some("WORK_GRAPH_BACKING_MEMORY"),
    );
}
