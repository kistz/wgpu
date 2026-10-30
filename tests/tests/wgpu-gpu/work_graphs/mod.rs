//! Tests of work graph execution.
//!
//! The work graph in `shader.hlsl` has a broadcasting entry node that launches 4 threads per
//! group of its dispatch grid. Each thread sends one record to a thread launch node, which adds
//! the entry record's value to `output[dispatch thread id]`.

use std::borrow::Cow;

use wgpu::{util::DeviceExt as _, Features};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        WORK_GRAPH_ENTRYPOINTS,
        WORK_GRAPH_DISPATCH,
        WORK_GRAPH_REPEATED_RECORD,
        WORK_GRAPH_MULTI_NODE_INPUT,
        WORK_GRAPH_SHARED_BACKING_MEMORY,
    ]);
}

const OUTPUT_LEN: usize = 16;

fn parameters() -> TestParameters {
    TestParameters::default()
        .features(Features::EXPERIMENTAL_WORK_GRAPHS | Features::PASSTHROUGH_SHADERS)
        .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        .limits(wgpu::Limits::downlevel_defaults())
}

/// An entry record with a dispatch grid of `groups` thread groups.
fn entry_record(groups: u32, value: u32) -> [u32; 4] {
    [groups, 1, 1, value]
}

fn input(records: &[[u32; 4]], record_stride: u64) -> wgpu::WorkGraphNodeInput<&[u8]> {
    wgpu::WorkGraphNodeInput {
        entrypoint_index: 0,
        num_records: records.len() as u32,
        record_stride,
        records: bytemuck::cast_slice(records),
    }
}

struct Resources {
    module: wgpu::ShaderModule,
    layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Resources {
    fn new(ctx: &TestingContext) -> Self {
        let module = unsafe {
            ctx.device
                .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("work graph library"),
                    hlsl: Some(Cow::Borrowed(include_str!("shader.hlsl"))),
                    ..Default::default()
                })
        };

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let output = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("output"),
                contents: bytemuck::cast_slice(&[0u32; OUTPUT_LEN]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });

        Self {
            module,
            layout,
            bind_group,
            output,
            readback,
        }
    }

    fn pipeline(&self, ctx: &TestingContext, program_name: &str) -> wgpu::WorkGraphPipeline {
        ctx.device
            .create_work_graph_pipeline(&wgpu::WorkGraphPipelineDescriptor {
                label: Some(program_name),
                layout: &self.layout,
                module: &self.module,
                program_name,
                entrypoints: &[],
                include_all_available_nodes: true,
            })
    }

    fn backing_memory(
        &self,
        ctx: &TestingContext,
        pipelines: &[&wgpu::WorkGraphPipeline],
    ) -> wgpu::Buffer {
        let size = pipelines
            .iter()
            .map(|pipeline| pipeline.memory_requirements().min_size)
            .max()
            .unwrap()
            .max(wgpu::WORK_GRAPH_BACKING_MEMORY_ALIGNMENT);
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("backing memory"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn begin_pass<'a>(&self, encoder: &'a mut wgpu::CommandEncoder) -> wgpu::ComputePass<'a> {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass
    }

    async fn submit_and_read(
        &self,
        ctx: &TestingContext,
        mut encoder: wgpu::CommandEncoder,
    ) -> Vec<u32> {
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, self.output.size());
        ctx.queue.submit(Some(encoder.finish()));

        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        let view = self.readback.slice(..).get_mapped_range().unwrap();
        let output = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        self.readback.unmap();
        output
    }
}

/// The expected output, given the value for each group of 4 threads.
fn expected(group_values: &[u32]) -> Vec<u32> {
    let mut output = vec![0; OUTPUT_LEN];
    for (group, &value) in group_values.iter().enumerate() {
        output[group * 4..][..4].fill(value);
    }
    output
}

#[apply(gpu_test!)]
static WORK_GRAPH_ENTRYPOINTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(|ctx| {
        let resources = Resources::new(&ctx);
        let pipeline = resources.pipeline(&ctx, "graph");

        let entrypoints = pipeline.entrypoints();
        assert_eq!(entrypoints.len(), 1);
        assert_eq!(entrypoints[0].node.name, "entry");
        assert_eq!(entrypoints[0].node.array_index, 0);
        assert_eq!(entrypoints[0].record_size, 16);
        assert_eq!(
            pipeline.entrypoint_index(&wgpu::WorkGraphNodeId {
                name: Cow::Borrowed("entry"),
                array_index: 0,
            }),
            Some(0)
        );
    });

#[apply(gpu_test!)]
static WORK_GRAPH_DISPATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);
        let pipeline = resources.pipeline(&ctx, "graph");
        let backing_memory = resources.backing_memory(&ctx, &[&pipeline]);

        let records = [entry_record(2, 1), entry_record(1, 10)];
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = resources.begin_pass(&mut encoder);
            pass.set_work_graph_pipeline(&pipeline, backing_memory.slice(..));
            unsafe { pass.dispatch_graph(&[input(&records, 16)]) };
        }

        let output = resources.submit_and_read(&ctx, encoder).await;
        assert_eq!(output, expected(&[11, 1]));
    });

#[apply(gpu_test!)]
static WORK_GRAPH_REPEATED_RECORD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);
        let pipeline = resources.pipeline(&ctx, "graph");
        let backing_memory = resources.backing_memory(&ctx, &[&pipeline]);

        let records = [entry_record(1, 100)];
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = resources.begin_pass(&mut encoder);
            pass.set_work_graph_pipeline(&pipeline, backing_memory.slice(..));
            unsafe {
                pass.dispatch_graph(&[wgpu::WorkGraphNodeInput {
                    num_records: 3,
                    ..input(&records, 0)
                }])
            };
        }

        let output = resources.submit_and_read(&ctx, encoder).await;
        assert_eq!(output, expected(&[300]));
    });

#[apply(gpu_test!)]
static WORK_GRAPH_MULTI_NODE_INPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);
        let pipeline = resources.pipeline(&ctx, "graph");
        let backing_memory = resources.backing_memory(&ctx, &[&pipeline]);

        let first = [entry_record(1, 1)];
        let second = [entry_record(2, 16)];
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = resources.begin_pass(&mut encoder);
            pass.set_work_graph_pipeline(&pipeline, backing_memory.slice(..));
            unsafe { pass.dispatch_graph(&[input(&first, 16), input(&second, 16)]) };
        }

        let output = resources.submit_and_read(&ctx, encoder).await;
        assert_eq!(output, expected(&[17, 16]));
    });

/// Uses one backing memory for repeated dispatches, for two work graphs, and across passes.
#[apply(gpu_test!)]
static WORK_GRAPH_SHARED_BACKING_MEMORY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(|ctx| async move {
        let resources = Resources::new(&ctx);
        let graph_a = resources.pipeline(&ctx, "graph_a");
        let graph_b = resources.pipeline(&ctx, "graph_b");
        let backing_memory = resources.backing_memory(&ctx, &[&graph_a, &graph_b]);

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = resources.begin_pass(&mut encoder);
            pass.set_work_graph_pipeline(&graph_a, backing_memory.slice(..));
            unsafe {
                pass.dispatch_graph(&[input(&[entry_record(1, 1)], 16)]);
                pass.dispatch_graph(&[input(&[entry_record(1, 2)], 16)]);
            }
            pass.set_work_graph_pipeline(&graph_b, backing_memory.slice(..));
            unsafe { pass.dispatch_graph(&[input(&[entry_record(1, 4)], 16)]) };
        }
        {
            let mut pass = resources.begin_pass(&mut encoder);
            pass.set_work_graph_pipeline(&graph_a, backing_memory.slice(..));
            unsafe { pass.dispatch_graph(&[input(&[entry_record(2, 8)], 16)]) };
        }

        let output = resources.submit_and_read(&ctx, encoder).await;
        assert_eq!(output, expected(&[15, 8]));
    });
