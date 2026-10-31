use crate::*;

/// Handle to a work graph pipeline.
///
/// A `WorkGraphPipeline` is a graph of node shaders from a passthrough shader library. It can be created with [`Device::create_work_graph_pipeline`] and is executed in a
/// [`ComputePass`] with [`ComputePass::set_work_graph_pipeline`] and
/// [`ComputePass::dispatch_graph`]. A work graph with mesh nodes is executed in a [`RenderPass`]
/// with [`RenderPass::set_work_graph_pipeline`] and [`RenderPass::dispatch_graph`] instead.
///
/// Requires [`Features::EXPERIMENTAL_WORK_GRAPHS`].
#[derive(Debug, Clone)]
pub struct WorkGraphPipeline {
    pub(crate) inner: dispatch::DispatchWorkGraphPipeline,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(WorkGraphPipeline: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(WorkGraphPipeline => .inner);

impl WorkGraphPipeline {
    /// Size requirements for the backing memory passed to
    /// [`ComputePass::set_work_graph_pipeline`].
    pub fn memory_requirements(&self) -> WorkGraphMemoryRequirements {
        self.inner.memory_requirements()
    }

    /// The entrypoints of the work graph, in entrypoint index order.
    pub fn entrypoints(&self) -> &[WorkGraphEntrypoint] {
        self.inner.entrypoints()
    }

    /// Returns the entrypoint index of `node`, or `None` if `node` is not an entrypoint.
    pub fn entrypoint_index(&self, node: &WorkGraphNodeId<'_>) -> Option<u32> {
        self.entrypoints()
            .iter()
            .position(|entrypoint| {
                entrypoint.node.name == node.name && entrypoint.node.array_index == node.array_index
            })
            .map(|index| index as u32)
    }

    #[cfg(custom)]
    /// Returns custom implementation of WorkGraphPipeline (if custom backend and is internally T)
    pub fn as_custom<T: custom::WorkGraphPipelineInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Describes a work graph pipeline.
///
/// For use with [`Device::create_work_graph_pipeline`].
#[derive(Clone, Debug)]
pub struct WorkGraphPipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline. It is shared by all nodes of the work graph.
    pub layout: &'a PipelineLayout,
    /// A shader module created with [`Device::create_shader_module_passthrough`] from a DXIL
    /// shader library, or from HLSL source that is compiled for a `lib_6_8` or newer target.
    /// On Vulkan, the shader library is SPIR-V that uses `SPV_AMDX_shader_enqueue`.
    pub module: &'a ShaderModule,
    /// Name of the work graph program. Vulkan ignores it.
    pub program_name: &'a str,
    /// Nodes that become entrypoints in addition to the ones the runtime determines from the
    /// shaders.
    pub entrypoints: &'a [WorkGraphNodeId<'a>],
    /// Adds every node shader in `module` to the work graph. Otherwise, only `entrypoints` and the
    /// nodes they output to are added.
    pub include_all_available_nodes: bool,
    /// Mesh nodes of the work graph, which draw in a render pass.
    ///
    /// Requires [`Features::EXPERIMENTAL_WORK_GRAPHS_1_1`]. All mesh nodes must render to the
    /// same attachment formats with the same sample count.
    pub mesh_nodes: &'a [WorkGraphMeshNode<'a>],
    /// Upper bounds on the input of each [`RenderPass::dispatch_graph`] call. Required if there are
    /// mesh nodes.
    pub max_input_records: Option<WorkGraphMaxInputRecords>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(WorkGraphPipelineDescriptor<'_>: Send, Sync);

/// Describes a mesh node of a work graph, and the graphics state that it draws with.
///
/// The mesh node shader is a node shader with `[NodeLaunch("mesh")]`. Like the fragment shader,
/// it is an export of the work graph's shader library.
#[derive(Clone, Debug)]
pub struct WorkGraphMeshNode<'a> {
    /// The ID of the node in the work graph.
    pub node: WorkGraphNodeId<'a>,
    /// Name of the mesh node shader in the shader library.
    pub mesh_entry_point: &'a str,
    /// The fragment shader and the color targets.
    pub fragment: Option<WorkGraphFragmentState<'a>>,
    /// The primitive state. `topology` must be the output topology of the mesh node shader.
    pub primitive: PrimitiveState,
    /// The depth and stencil state.
    pub depth_stencil: Option<DepthStencilState>,
    /// The multisample state.
    pub multisample: MultisampleState,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(WorkGraphMeshNode<'_>: Send, Sync);

/// Describes the fragment shader of a work graph mesh node.
#[derive(Clone, Debug)]
pub struct WorkGraphFragmentState<'a> {
    /// Name of the fragment shader in the shader library.
    pub entry_point: &'a str,
    /// The color targets that the fragment shader writes to.
    pub targets: &'a [Option<ColorTargetState>],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(WorkGraphFragmentState<'_>: Send, Sync);
