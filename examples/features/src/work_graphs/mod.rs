use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

const SNOWFLAKE: u32 = 0;
const CARPET: u32 = 1;

/// The input record of the `Entry` node.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Fractal {
    center: [f32; 2],
    radius: f32,
    rotation: f32,
    kind: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Scene {
    focus: [f32; 2],
    focus_radius: f32,
    _padding: f32,
}

struct Example {
    graph: wgpu::WorkGraphPipeline,
    entrypoint_index: u32,
    backing_memory: wgpu::Buffer,
    scene_buffer: wgpu::Buffer,
    graph_bind_group_layout: wgpu::BindGroupLayout,
    graph_bind_group: wgpu::BindGroup,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    canvas: wgpu::Texture,
    frame: u32,
}

impl Example {
    fn create_canvas(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        graph_bind_group_layout: &wgpu::BindGroupLayout,
        blit_bind_group_layout: &wgpu::BindGroupLayout,
        scene_buffer: &wgpu::Buffer,
    ) -> (wgpu::Texture, wgpu::BindGroup, wgpu::BindGroup) {
        let canvas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = canvas.create_view(&wgpu::TextureViewDescriptor::default());

        let graph_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("work graph"),
            layout: graph_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene_buffer.as_entire_binding(),
                },
            ],
        });
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: blit_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        (canvas, graph_bind_group, blit_bind_group)
    }

    /// Places the fractals side by side, or above each other on a portrait canvas.
    fn fractals(&self, time: f32) -> [Fractal; 2] {
        let width = self.canvas.width() as f32;
        let height = self.canvas.height() as f32;
        let (centers, radius) = if width >= height {
            (
                [[width * 0.27, height * 0.5], [width * 0.73, height * 0.5]],
                (width * 0.22).min(height * 0.42),
            )
        } else {
            (
                [[width * 0.5, height * 0.27], [width * 0.5, height * 0.73]],
                (width * 0.42).min(height * 0.22),
            )
        };

        [
            Fractal {
                center: centers[0],
                radius,
                rotation: time * 0.25,
                kind: SNOWFLAKE,
            },
            Fractal {
                center: centers[1],
                radius,
                rotation: time * -0.2,
                kind: CARPET,
            },
        ]
    }

    fn scene(&self, time: f32) -> Scene {
        let width = self.canvas.width() as f32;
        let height = self.canvas.height() as f32;
        Scene {
            focus: [
                width * (0.5 + 0.38 * (time * 0.5).sin()),
                height * (0.5 + 0.25 * (time * 0.9).sin()),
            ],
            focus_radius: width.min(height) * 0.35,
            _padding: 0.0,
        }
    }
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_WORK_GRAPHS
            | wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::CLEAR_TEXTURE
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let library = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("work graph library"),
                hlsl: Some(Cow::Borrowed(include_str!("shader.hlsl"))),
                spirv: Some(wgpu::util::make_spirv_raw(include_bytes!("shader.spv"))),
                ..Default::default()
            })
        };

        // The HLSL registers follow from this layout: the storage texture is `u0` and the
        // uniform buffer is `b0`.
        let graph_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("work graph"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_of::<Scene>() as u64),
                        },
                        count: None,
                    },
                ],
            });
        let graph_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("work graph"),
            bind_group_layouts: &[Some(&graph_bind_group_layout)],
            immediate_size: 0,
        });

        let graph = device.create_work_graph_pipeline(&wgpu::WorkGraphPipelineDescriptor {
            label: Some("fractals"),
            layout: &graph_layout,
            module: &library,
            program_name: "fractals",
            entrypoints: &[],
            include_all_available_nodes: true,
            mesh_nodes: &[],
            max_input_records: None,
        });
        let entrypoint_index = graph
            .entrypoint_index(&wgpu::WorkGraphNodeId {
                name: Cow::Borrowed("Entry"),
                array_index: 0,
            })
            .unwrap();

        // The backing memory holds the records while the graph runs.
        let backing_memory = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("work graph backing memory"),
            size: graph
                .memory_requirements()
                .min_size
                .max(wgpu::WORK_GRAPH_BACKING_MEMORY_ALIGNMENT),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene"),
            size: size_of::<Scene>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });
        let blit_shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (canvas, graph_bind_group, blit_bind_group) = Self::create_canvas(
            config,
            device,
            &graph_bind_group_layout,
            &blit_bind_group_layout,
            &scene_buffer,
        );

        Example {
            graph,
            entrypoint_index,
            backing_memory,
            scene_buffer,
            graph_bind_group_layout,
            graph_bind_group,
            blit_bind_group_layout,
            blit_bind_group,
            blit_pipeline,
            canvas,
            frame: 0,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        (self.canvas, self.graph_bind_group, self.blit_bind_group) = Self::create_canvas(
            config,
            device,
            &self.graph_bind_group_layout,
            &self.blit_bind_group_layout,
            &self.scene_buffer,
        );
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let time = self.frame as f32 / 60.0;
        self.frame += 1;
        queue.write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&self.scene(time)));
        let fractals = self.fractals(time);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_texture(&self.canvas, &wgpu::ImageSubresourceRange::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_bind_group(0, &self.graph_bind_group, &[]);
            cpass.set_work_graph_pipeline(&self.graph, self.backing_memory.slice(..));
            // SAFETY: The nodes limit the number of records and dispatch grids that they output.
            unsafe {
                cpass.dispatch_graph(&[wgpu::WorkGraphNodeInput {
                    entrypoint_index: self.entrypoint_index,
                    num_records: fractals.len() as u32,
                    record_stride: size_of::<Fractal>() as u64,
                    records: bytemuck::cast_slice(&fractals),
                }]);
            }
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("work_graphs");
}

#[cfg(test)]
#[wgpu_test::apply(wgpu_test::gpu_test!)]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "work_graphs",
    image_path: "/examples/features/src/work_graphs/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::empty(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.01)],
    _phantom: std::marker::PhantomData::<Example>,
};
