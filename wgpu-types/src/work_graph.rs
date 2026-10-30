use alloc::borrow::Cow;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::BufferAddress;

/// Required alignment of the offset of a work graph's backing memory.
pub const WORK_GRAPH_BACKING_MEMORY_ALIGNMENT: BufferAddress = 8;

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
