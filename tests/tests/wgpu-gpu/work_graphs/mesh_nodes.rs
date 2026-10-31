//! Tests of work graphs with mesh nodes.
//!
//! The work graph in `mesh_nodes.hlsl` has a thread launch entry node that passes its record
//! to a mesh node, which draws a triangle covering the render target in the record's color.

use std::borrow::Cow;

use wgpu::Features;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, TestParameters};

const SIZE: u32 = 4;

#[apply(gpu_test!)]
pub static WORK_GRAPH_MESH_NODES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::EXPERIMENTAL_WORK_GRAPHS
                    | Features::EXPERIMENTAL_WORK_GRAPHS_1_1
                    | Features::PASSTHROUGH_SHADERS,
            )
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let module = unsafe {
            ctx.device
                .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("mesh node library"),
                    hlsl: Some(Cow::Borrowed(include_str!("mesh_nodes.hlsl"))),
                    ..Default::default()
                })
        };
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_work_graph_pipeline(&wgpu::WorkGraphPipelineDescriptor {
                label: Some("mesh nodes"),
                layout: &layout,
                module: &module,
                program_name: "graph",
                entrypoints: &[wgpu::WorkGraphNodeId {
                    name: Cow::Borrowed("entry"),
                    array_index: 0,
                }],
                include_all_available_nodes: false,
                mesh_nodes: &[wgpu::WorkGraphMeshNode {
                    node: wgpu::WorkGraphNodeId {
                        name: Cow::Borrowed("triangle"),
                        array_index: 0,
                    },
                    mesh_entry_point: "triangle_mesh",
                    fragment: Some(wgpu::WorkGraphFragmentState {
                        entry_point: "triangle_fragment",
                        targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                }],
                max_input_records: Some(wgpu::WorkGraphMaxInputRecords {
                    records: 1,
                    node_inputs: 1,
                }),
            });
        let backing_memory = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("backing memory"),
            size: pipeline
                .memory_requirements()
                .min_size
                .max(wgpu::WORK_GRAPH_BACKING_MEMORY_ALIGNMENT),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let color = [0.0f32, 1.0, 0.0, 1.0];
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_work_graph_pipeline(&pipeline, backing_memory.slice(..));
            unsafe {
                pass.dispatch_graph(&[wgpu::WorkGraphNodeInput {
                    entrypoint_index: 0,
                    num_records: 1,
                    record_stride: 16,
                    records: bytemuck::cast_slice(&color),
                }])
            };
        }
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            target.size(),
        );
        ctx.queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let data = readback.slice(..).get_mapped_range().unwrap();
        for row in data.chunks(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize) {
            for pixel in row[..(SIZE * 4) as usize].chunks(4) {
                assert_eq!(pixel, [0, 255, 0, 255]);
            }
        }
    });
