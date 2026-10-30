use crate::*;

/// Handle to a work graph pipeline.
///
/// A `WorkGraphPipeline` is a graph of node shaders from a passthrough DXIL or HLSL shader
/// library. It can be created with [`Device::create_work_graph_pipeline`] and is executed in a
/// [`ComputePass`] with [`ComputePass::set_work_graph_pipeline`] and
/// [`ComputePass::dispatch_graph`].
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
    pub module: &'a ShaderModule,
    /// Name of the work graph program.
    pub program_name: &'a str,
    /// Nodes that become entrypoints in addition to the ones the runtime determines from the
    /// shaders.
    pub entrypoints: &'a [WorkGraphNodeId<'a>],
    /// Adds every node shader in `module` to the work graph. Otherwise, only `entrypoints` and the
    /// nodes they output to are added.
    pub include_all_available_nodes: bool,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(WorkGraphPipelineDescriptor<'_>: Send, Sync);
