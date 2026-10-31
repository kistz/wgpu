use alloc::borrow::Cow;

use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::BufferAddress;

/// Required alignment of the offset of a work graph's backing memory.
pub const WORK_GRAPH_BACKING_MEMORY_ALIGNMENT: BufferAddress = 64;

/// Required alignment of the offset of work graph input in GPU memory.
pub const WORK_GRAPH_GPU_INPUT_ALIGNMENT: BufferAddress = 8;

/// Identifies a node of a work graph by its name and its index in the node array.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkGraphNodeId<'a> {
    /// Name of the node.
    pub name: Cow<'a, str>,
    /// Index of the node in its node array.
    pub array_index: u32,
}

impl WorkGraphNodeId<'_> {
    /// Returns a copy of this id that owns its name.
    #[must_use]
    pub fn into_owned(self) -> WorkGraphNodeId<'static> {
        WorkGraphNodeId {
            name: Cow::Owned(self.name.into_owned()),
            array_index: self.array_index,
        }
    }
}

/// Size requirements for the backing memory of a work graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkGraphMemoryRequirements {
    /// Smallest backing memory size with which the work graph can execute.
    pub min_size: BufferAddress,
    /// Size above which more backing memory does not improve performance.
    pub max_size: BufferAddress,
    /// Sizes above `min_size` should exceed it by a multiple of this value.
    pub size_granularity: BufferAddress,
}

/// Upper bounds on the input of a single dispatch of a work graph with mesh nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkGraphMaxInputRecords {
    /// Maximum number of records summed over all inputs of a dispatch.
    pub records: u32,
    /// Maximum number of inputs of a dispatch.
    pub node_inputs: u32,
}

/// An entrypoint of a work graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkGraphEntrypoint {
    /// The node that receives the input records.
    pub node: WorkGraphNodeId<'static>,
    /// Size of one input record in bytes. This is zero if the node has no input record.
    pub record_size: u32,
    /// Required alignment of input records in bytes.
    pub record_alignment: u32,
    /// The value of [`WorkGraphGpuNodeInput::gpu_input_index`] that refers to this entrypoint.
    ///
    /// This is the entrypoint index on DX12, and the node index that the driver assigns on
    /// Vulkan.
    pub gpu_input_index: u32,
}

impl WorkGraphEntrypoint {
    /// Required alignment of the stride between input records.
    #[must_use]
    pub fn record_stride_alignment(&self) -> u64 {
        u64::from(self.record_alignment.max(4))
    }

    /// Number of bytes of record data read for `num_records` records with `record_stride`.
    #[must_use]
    pub fn required_records_size(&self, num_records: u32, record_stride: u64) -> u64 {
        if num_records == 0 || self.record_size == 0 {
            0
        } else if record_stride == 0 {
            u64::from(self.record_size)
        } else {
            u64::from(num_records - 1) * record_stride + u64::from(self.record_size)
        }
    }
}

/// Input records for one entrypoint of a work graph.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkGraphNodeInput<R> {
    /// Index of the entrypoint in the entrypoints of the work graph pipeline.
    pub entrypoint_index: u32,
    /// Number of records to submit to the entrypoint.
    pub num_records: u32,
    /// Distance in bytes between the starts of consecutive records in `records`.
    ///
    /// If this is zero, the first record is submitted `num_records` times.
    pub record_stride: u64,
    /// The record data.
    pub records: R,
}

/// The layout of work graph input in GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum WorkGraphGpuInputMode {
    /// The input is a [`WorkGraphGpuNodeInput`].
    SingleNode,
    /// The input is a [`WorkGraphGpuMultiNodeInput`].
    MultiNode,
}

impl WorkGraphGpuInputMode {
    /// Size in bytes of the input.
    #[must_use]
    pub const fn input_size(self) -> BufferAddress {
        match self {
            Self::SingleNode => size_of::<WorkGraphGpuNodeInput>() as BufferAddress,
            Self::MultiNode => size_of::<WorkGraphGpuMultiNodeInput>() as BufferAddress,
        }
    }
}

/// GPU memory layout of input records for one entrypoint of a work graph.
///
/// The address of this struct must be a multiple of [`WORK_GRAPH_GPU_INPUT_ALIGNMENT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct WorkGraphGpuNodeInput {
    /// The [`WorkGraphEntrypoint::gpu_input_index`] of the entrypoint.
    pub gpu_input_index: u32,
    /// Number of records to submit to the entrypoint.
    pub num_records: u32,
    /// GPU address of the first record.
    ///
    /// The address and `record_stride` must be multiples of the entrypoint's
    /// [`WorkGraphEntrypoint::record_stride_alignment`].
    pub records_address: u64,
    /// Distance in bytes between the starts of consecutive records.
    ///
    /// If this is zero, the first record is submitted `num_records` times. Vulkan does not
    /// support zero. Otherwise it must be at least the entrypoint's
    /// [`WorkGraphEntrypoint::record_size`].
    pub record_stride: u64,
}

impl WorkGraphGpuNodeInput {
    /// Returns the bytes representation of the struct, ready to be written into a buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// GPU memory layout of input records for several entrypoints of a work graph.
///
/// The address of this struct must be a multiple of [`WORK_GRAPH_GPU_INPUT_ALIGNMENT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct WorkGraphGpuMultiNodeInput {
    /// Number of [`WorkGraphGpuNodeInput`]s.
    pub num_node_inputs: u32,
    _padding: u32,
    /// GPU address of the first [`WorkGraphGpuNodeInput`].
    ///
    /// The address and `node_inputs_stride` must be multiples of
    /// [`WORK_GRAPH_GPU_INPUT_ALIGNMENT`].
    pub node_inputs_address: u64,
    /// Distance in bytes between the starts of consecutive [`WorkGraphGpuNodeInput`]s.
    ///
    /// If this is zero, the first node input is used `num_node_inputs` times. Vulkan does not
    /// support zero. Otherwise it must be at least the size of [`WorkGraphGpuNodeInput`].
    pub node_inputs_stride: u64,
}

impl WorkGraphGpuMultiNodeInput {
    /// Creates a multi node input.
    #[must_use]
    pub const fn new(
        num_node_inputs: u32,
        node_inputs_address: u64,
        node_inputs_stride: u64,
    ) -> Self {
        Self {
            num_node_inputs,
            _padding: 0,
            node_inputs_address,
            node_inputs_stride,
        }
    }

    /// Returns the bytes representation of the struct, ready to be written into a buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
