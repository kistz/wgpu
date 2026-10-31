//! Declarations of `VK_AMDX_shader_enqueue` revision 2.
//!
//! `ash` only has revision 1, whose commands and structures differ. The layouts match
//! `vulkan_beta.h` of Vulkan SDK 1.4.328.

use core::{
    ffi::{c_void, CStr},
    mem, ptr,
};

use ash::vk;

pub const NAME: &CStr = vk::AMDX_SHADER_ENQUEUE_NAME;
pub const SPEC_VERSION: u32 = 2;

/// `VkPhysicalDeviceShaderEnqueueFeaturesAMDX`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PhysicalDeviceShaderEnqueueFeatures {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub shader_enqueue: vk::Bool32,
    pub shader_mesh_enqueue: vk::Bool32,
}

impl Default for PhysicalDeviceShaderEnqueueFeatures {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES_AMDX,
            p_next: ptr::null_mut(),
            shader_enqueue: vk::FALSE,
            shader_mesh_enqueue: vk::FALSE,
        }
    }
}

unsafe impl Send for PhysicalDeviceShaderEnqueueFeatures {}
unsafe impl Sync for PhysicalDeviceShaderEnqueueFeatures {}
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceShaderEnqueueFeatures {}
unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceShaderEnqueueFeatures {}

/// `VkPhysicalDeviceShaderEnqueuePropertiesAMDX`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PhysicalDeviceShaderEnqueueProperties {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub max_execution_graph_depth: u32,
    pub max_execution_graph_shader_output_nodes: u32,
    pub max_execution_graph_shader_payload_size: u32,
    pub max_execution_graph_shader_payload_count: u32,
    pub execution_graph_dispatch_address_alignment: u32,
    pub max_execution_graph_workgroup_count: [u32; 3],
    pub max_execution_graph_workgroups: u32,
}

impl Default for PhysicalDeviceShaderEnqueueProperties {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES_AMDX,
            p_next: ptr::null_mut(),
            max_execution_graph_depth: 0,
            max_execution_graph_shader_output_nodes: 0,
            max_execution_graph_shader_payload_size: 0,
            max_execution_graph_shader_payload_count: 0,
            execution_graph_dispatch_address_alignment: 0,
            max_execution_graph_workgroup_count: [0; 3],
            max_execution_graph_workgroups: 0,
        }
    }
}

unsafe impl Send for PhysicalDeviceShaderEnqueueProperties {}
unsafe impl Sync for PhysicalDeviceShaderEnqueueProperties {}
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for PhysicalDeviceShaderEnqueueProperties {}

/// `VkExecutionGraphPipelineScratchSizeAMDX`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecutionGraphPipelineScratchSize {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub min_size: vk::DeviceSize,
    pub max_size: vk::DeviceSize,
    pub size_granularity: vk::DeviceSize,
}

impl Default for ExecutionGraphPipelineScratchSize {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE_AMDX,
            p_next: ptr::null_mut(),
            min_size: 0,
            max_size: 0,
            size_granularity: 0,
        }
    }
}

type CreateExecutionGraphPipelines = unsafe extern "system" fn(
    device: vk::Device,
    pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    create_infos: *const vk::ExecutionGraphPipelineCreateInfoAMDX<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    pipelines: *mut vk::Pipeline,
) -> vk::Result;
type GetExecutionGraphPipelineScratchSize = unsafe extern "system" fn(
    device: vk::Device,
    execution_graph: vk::Pipeline,
    size_info: *mut ExecutionGraphPipelineScratchSize,
) -> vk::Result;
type GetExecutionGraphPipelineNodeIndex = unsafe extern "system" fn(
    device: vk::Device,
    execution_graph: vk::Pipeline,
    node_info: *const vk::PipelineShaderStageNodeCreateInfoAMDX<'_>,
    node_index: *mut u32,
) -> vk::Result;
type CmdInitializeGraphScratchMemory = unsafe extern "system" fn(
    command_buffer: vk::CommandBuffer,
    execution_graph: vk::Pipeline,
    scratch: vk::DeviceAddress,
    scratch_size: vk::DeviceSize,
);
type CmdDispatchGraph = unsafe extern "system" fn(
    command_buffer: vk::CommandBuffer,
    scratch: vk::DeviceAddress,
    scratch_size: vk::DeviceSize,
    count_info: *const vk::DispatchGraphCountInfoAMDX,
);
type CmdDispatchGraphIndirectCount = unsafe extern "system" fn(
    command_buffer: vk::CommandBuffer,
    scratch: vk::DeviceAddress,
    scratch_size: vk::DeviceSize,
    count_info: vk::DeviceAddress,
);

/// Casts a loaded command to its signature.
unsafe fn cast<F: Copy>(function: unsafe extern "system" fn()) -> F {
    const { assert!(size_of::<F>() == size_of::<unsafe extern "system" fn()>()) };
    unsafe { mem::transmute_copy(&function) }
}

/// Device-level commands of `VK_AMDX_shader_enqueue`.
pub struct DeviceFn {
    device: vk::Device,
    create_execution_graph_pipelines: CreateExecutionGraphPipelines,
    get_execution_graph_pipeline_scratch_size: GetExecutionGraphPipelineScratchSize,
    get_execution_graph_pipeline_node_index: GetExecutionGraphPipelineNodeIndex,
    cmd_initialize_graph_scratch_memory: CmdInitializeGraphScratchMemory,
    cmd_dispatch_graph: CmdDispatchGraph,
    cmd_dispatch_graph_indirect: CmdDispatchGraph,
    cmd_dispatch_graph_indirect_count: CmdDispatchGraphIndirectCount,
}

impl DeviceFn {
    /// Loads the commands, or returns `None` if the device lacks any of them.
    pub fn load(instance: &ash::Instance, device: &ash::Device) -> Option<Self> {
        let handle = device.handle();
        let load = |name: &CStr| unsafe { instance.get_device_proc_addr(handle, name.as_ptr()) };
        unsafe {
            Some(Self {
                device: handle,
                create_execution_graph_pipelines: cast(load(
                    c"vkCreateExecutionGraphPipelinesAMDX",
                )?),
                get_execution_graph_pipeline_scratch_size: cast(load(
                    c"vkGetExecutionGraphPipelineScratchSizeAMDX",
                )?),
                get_execution_graph_pipeline_node_index: cast(load(
                    c"vkGetExecutionGraphPipelineNodeIndexAMDX",
                )?),
                cmd_initialize_graph_scratch_memory: cast(load(
                    c"vkCmdInitializeGraphScratchMemoryAMDX",
                )?),
                cmd_dispatch_graph: cast(load(c"vkCmdDispatchGraphAMDX")?),
                cmd_dispatch_graph_indirect: cast(load(c"vkCmdDispatchGraphIndirectAMDX")?),
                cmd_dispatch_graph_indirect_count: cast(load(
                    c"vkCmdDispatchGraphIndirectCountAMDX",
                )?),
            })
        }
    }

    pub unsafe fn create_execution_graph_pipeline(
        &self,
        create_info: &vk::ExecutionGraphPipelineCreateInfoAMDX<'_>,
    ) -> Result<vk::Pipeline, vk::Result> {
        let mut pipeline = vk::Pipeline::null();
        unsafe {
            (self.create_execution_graph_pipelines)(
                self.device,
                vk::PipelineCache::null(),
                1,
                create_info,
                ptr::null(),
                &mut pipeline,
            )
        }
        .result_with_success(pipeline)
    }

    pub unsafe fn get_execution_graph_pipeline_scratch_size(
        &self,
        execution_graph: vk::Pipeline,
    ) -> Result<ExecutionGraphPipelineScratchSize, vk::Result> {
        let mut size = ExecutionGraphPipelineScratchSize::default();
        unsafe {
            (self.get_execution_graph_pipeline_scratch_size)(
                self.device,
                execution_graph,
                &mut size,
            )
        }
        .result_with_success(size)
    }

    pub unsafe fn get_execution_graph_pipeline_node_index(
        &self,
        execution_graph: vk::Pipeline,
        node_info: &vk::PipelineShaderStageNodeCreateInfoAMDX<'_>,
    ) -> Result<u32, vk::Result> {
        let mut node_index = 0;
        unsafe {
            (self.get_execution_graph_pipeline_node_index)(
                self.device,
                execution_graph,
                node_info,
                &mut node_index,
            )
        }
        .result_with_success(node_index)
    }

    pub unsafe fn cmd_initialize_graph_scratch_memory(
        &self,
        command_buffer: vk::CommandBuffer,
        execution_graph: vk::Pipeline,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
    ) {
        unsafe {
            (self.cmd_initialize_graph_scratch_memory)(
                command_buffer,
                execution_graph,
                scratch,
                scratch_size,
            )
        }
    }

    pub unsafe fn cmd_dispatch_graph(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: &vk::DispatchGraphCountInfoAMDX,
    ) {
        unsafe { (self.cmd_dispatch_graph)(command_buffer, scratch, scratch_size, count_info) }
    }

    pub unsafe fn cmd_dispatch_graph_indirect(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: &vk::DispatchGraphCountInfoAMDX,
    ) {
        unsafe {
            (self.cmd_dispatch_graph_indirect)(command_buffer, scratch, scratch_size, count_info)
        }
    }

    pub unsafe fn cmd_dispatch_graph_indirect_count(
        &self,
        command_buffer: vk::CommandBuffer,
        scratch: vk::DeviceAddress,
        scratch_size: vk::DeviceSize,
        count_info: vk::DeviceAddress,
    ) {
        unsafe {
            (self.cmd_dispatch_graph_indirect_count)(
                command_buffer,
                scratch,
                scratch_size,
                count_info,
            )
        }
    }
}
