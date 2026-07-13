use bytemuck::{Pod, Zeroable};
use std::f32::consts;
use wgpu::{hal::DynDevice, util::DeviceExt, RenderPipeline};

fn compile_hlsl(device: &wgpu::Device, entry: &str, stage_str: &str) -> wgpu::ShaderModule {
    let out_path = format!(
        "{}/src/mesh_shader/shader.{stage_str}.cso",
        env!("CARGO_MANIFEST_DIR")
    );
    let cmd = std::process::Command::new("dxc")
        .args([
            "-T",
            &format!("{stage_str}_6_5"),
            "-E",
            entry,
            &format!("{}/src/mesh_shader/shader.hlsl", env!("CARGO_MANIFEST_DIR")),
            "-Fo",
            &out_path,
        ])
        .output()
        .unwrap();
    if !cmd.status.success() {
        panic!("DXC failed:\n{}", String::from_utf8(cmd.stderr).unwrap());
    }
    let file = std::fs::read(&out_path).unwrap();
    std::fs::remove_file(out_path).unwrap();
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            dxil: Some(std::borrow::Cow::Owned(file)),
            ..Default::default()
        })
    }
}

struct Example {
    /* vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    pipeline_wire: Option<wgpu::RenderPipeline>, */
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 10.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(1.5f32, -5.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        projection * view
    }
}

impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        unsafe {
            /* device.as_hal_mut(|h: Option<&mut wgpu::hal::dx12::CommandEncoder>| unsafe {
                h.unwrap().draw_graph()
            }); */
            let what = device.as_hal::<wgpu::hal::api::Dx12>().unwrap();
            //what.state_object();
        }
        // Create the vertex and index buffers
        /* let vertex_size = size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Create the texture
        let size = 256u32;
        let texels = create_texels(size as usize);
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        queue.write_texture(
            texture.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: None,
            },
            texture_extent,
        );

        // Create other resources
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
            label: None,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 4 * 4,
                    shader_location: 1,
                },
            ],
        })];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let pipeline_wire = if device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            let pipeline_wire = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_wire"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.view_formats[0],
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                operation: wgpu::BlendOperation::Add,
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Line,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            Some(pipeline_wire)
        } else {
            None
        };

        // Done
        Example {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
            pipeline_wire,
        } */

        Example {}
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        /* let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref)); */
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            unsafe {
                encoder.as_hal_mut(|h: Option<&mut wgpu::hal::dx12::CommandEncoder>| unsafe {
                    //h.unwrap().draw_graph()

                    // For production, load your compiled DXIL bytecode (.bin) file containing SM 6.8
                    let dxil_bytecode: Vec<u8> =
                        std::fs::read("work_graph.dxil").expect("Failed to load DXIL");

                    // -------------------------------------------------------------
                    // 3. Define the State Object Subobjects to assemble the Work Graph
                    // -------------------------------------------------------------
                    let mut subobjects = Vec::new();

                    // A. Define the DXIL Library container
                    let dxil_lib_desc = D3D12_DXIL_LIBRARY_DESC {
                        DXILLibrary: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: dxil_bytecode.as_ptr() as *const _,
                            BytecodeLength: dxil_bytecode.len(),
                        },
                        NumExports: 0, // 0 exports means implicitly export all nodes in the file
                        pExports: std::ptr::null(),
                    };

                    subobjects.push(D3D12_STATE_SUBOBJECT {
                        Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
                        pDesc: &dxil_lib_desc as *const _ as *const _,
                    });

                    // B. Explicitly define the Work Graph Config
                    let graph_name = windows::core::w!("MyWorkGraph");
                    let work_graph_desc = D3D12_WORK_GRAPH_DESC {
                        ProgramName: graph_name.as_ptr(),
                        Flags: D3D12_WORK_GRAPH_FLAG_NONE,
                        NumExplicitNodes: 0, // Auto-detect nodes based on shader entry point attributes
                        pExplicitNodes: std::ptr::null(),
                    };

                    subobjects.push(D3D12_STATE_SUBOBJECT {
                        Type: D3D12_STATE_SUBOBJECT_TYPE_WORK_GRAPH,
                        pDesc: &work_graph_desc as *const _ as *const _,
                    });

                    // -------------------------------------------------------------
                    // 4. Create the final Executable State Object
                    // -------------------------------------------------------------
                    let state_object_desc = D3D12_STATE_OBJECT_DESC {
                        Type: D3D12_STATE_OBJECT_TYPE_COLLECTION,
                        NumSubobjects: subobjects.len() as u32,
                        pSubobjects: subobjects.as_ptr(),
                    };

                    let mut state_object: Option<ID3D12StateObject> = None;
                    device.CreateStateObject(&state_object_desc, &mut state_object)?;
                    let state_object = state_object.unwrap();

                    // -------------------------------------------------------------
                    // 5. Query Graph Properties and Allocate Backing Memory
                    // -------------------------------------------------------------
                    // Work Graphs require scratch memory allocated by the CPU for internal GPU scheduling data
                    let work_graph_properties: ID3D12WorkGraphProperties = state_object.cast()?;
                    let graph_index = work_graph_properties.GetWorkGraphIndex(graph_name);

                    let mut memory_requirements = D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS::default();
                    work_graph_properties
                        .GetWorkGraphMemoryRequirements(graph_index, &mut memory_requirements);

                    // Allocate a raw GPU buffer matched exactly to 'memory_requirements.MaxSizeInBytes'
                    let backing_memory_buffer =
                        create_gpu_buffer(device, memory_requirements.MaxSizeInBytes)?;

                    Ok(())
                });
            }
            /* let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.push_debug_group("Prepare data for draw.");
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.bind_group, &[]);
                rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                rpass.pop_debug_group();
                rpass.insert_debug_marker("Draw!");
                rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
                if let Some(ref pipe) = self.pipeline_wire {
                    rpass.set_pipeline(pipe);
                    rpass.draw_indexed(0..self.index_count as u32, 0, 0..1);
                }
            }

            queue.submit(Some(encoder.finish())); */
        }
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_WORK_GRAPHS | wgpu::Features::PASSTHROUGH_SHADERS
    }
}

pub fn main() {
    crate::framework::run::<Example>("work-graphs");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "cube",
    // Generated on 1080ti on Vk/Windows
    image_path: "/examples/features/src/cube/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default(),
    comparisons: &[
        wgpu_test::ComparisonType::Mean(0.041), // Bounded by Apple A9
    ],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST_LINES: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "cube-lines",
    // Generated on 1080ti on Vk/Windows
    image_path: "/examples/features/src/cube/screenshot-lines.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::POLYGON_MODE_LINE,
    base_test_parameters: wgpu_test::TestParameters::default(),
    // We're looking for tiny changes here, so we focus on a spike in the 95th percentile.
    comparisons: &[
        wgpu_test::ComparisonType::Mean(0.05), // Bounded by Intel 630 on Vk/Windows
        wgpu_test::ComparisonType::Percentile {
            percentile: 0.95,
            threshold: 0.36,
        }, // Bounded by 1080ti on DX12
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
