//! Tests of [`wgpu::WorkGraphPipeline`] validation.
//!
//! The noop backend reports a minimum backing memory size of 256 bytes, and one
//! entrypoint for each node listed in [`WorkGraphPipelineDescriptor::entrypoints`],
//! each with a record size of 16 bytes and a record alignment of 4 bytes.

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
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
    });
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
