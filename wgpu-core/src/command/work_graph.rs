//! Work graph state and validation shared by compute and render passes.

use alloc::{sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress,
};

use crate::{pipeline::WorkGraphPipeline, resource::Buffer};

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum WorkGraphError {
    #[error("Work graph pipeline must be set")]
    MissingPipeline,
    #[error(
        "Backing memory offset {0} is not a multiple of {alignment}",
        alignment = wgt::WORK_GRAPH_BACKING_MEMORY_ALIGNMENT
    )]
    UnalignedBackingMemoryOffset(BufferAddress),
    #[error("Backing memory of {size} bytes starting at offset {offset} would overrun buffer of size {buffer_size}")]
    BackingMemoryOverrun {
        offset: BufferAddress,
        size: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error("Backing memory of {size} bytes is smaller than the minimum size {min_size} required by the work graph")]
    BackingMemoryTooSmall {
        size: BufferAddress,
        min_size: BufferAddress,
    },
    #[error("Entrypoint index {index} of input {input} is out of range for a work graph with {count} entrypoints")]
    InvalidEntrypoint {
        input: usize,
        index: u32,
        count: usize,
    },
    #[error(
        "Record stride {stride} of input {input} is smaller than the record size {record_size}"
    )]
    RecordStrideTooSmall {
        input: usize,
        stride: u64,
        record_size: u32,
    },
    #[error("Record stride {stride} of input {input} is not a multiple of {alignment}")]
    UnalignedRecordStride {
        input: usize,
        stride: u64,
        alignment: u64,
    },
    #[error("Input {input} provides {size} bytes of records, but {required} bytes are required")]
    RecordsTooSmall {
        input: usize,
        size: u64,
        required: u64,
    },
    #[error(
        "GPU input offset {0} is not a multiple of {alignment}",
        alignment = wgt::WORK_GRAPH_GPU_INPUT_ALIGNMENT
    )]
    UnalignedGpuInputOffset(BufferAddress),
    #[error("GPU input of {input_size} bytes starting at offset {offset} would overrun buffer of size {buffer_size}")]
    GpuInputOverrun {
        input_size: BufferAddress,
        offset: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error("Work graph pipelines with mesh nodes can only be used in a render pass")]
    MeshNodesOutsideRenderPass,
    #[error("Work graph pipelines without mesh nodes can only be used in a compute pass")]
    NoMeshNodesInRenderPass,
    #[error("The dispatch has {records} records in {node_inputs} inputs, but the work graph pipeline allows at most {max_records} records in {max_node_inputs} inputs")]
    TooManyInputRecords {
        records: u64,
        node_inputs: usize,
        max_records: u32,
        max_node_inputs: u32,
    },
}

impl WebGpuError for WorkGraphError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

pub(super) struct WorkGraphState {
    pub(super) pipeline: Arc<WorkGraphPipeline>,
    pub(super) backing_memory: Arc<Buffer>,
    pub(super) offset: BufferAddress,
    pub(super) size: BufferAddress,
    /// The backing memory must be initialized by the next dispatch.
    pub(super) initialize: bool,
}

/// Validates the backing memory region of `pipeline` and returns its size.
pub(super) fn validate_backing_memory(
    pipeline: &WorkGraphPipeline,
    backing_memory: &Buffer,
    offset: BufferAddress,
    size: Option<BufferAddress>,
) -> Result<BufferAddress, WorkGraphError> {
    if !offset.is_multiple_of(wgt::WORK_GRAPH_BACKING_MEMORY_ALIGNMENT) {
        return Err(WorkGraphError::UnalignedBackingMemoryOffset(offset));
    }
    let buffer_size = backing_memory.size;
    let size = size.unwrap_or(buffer_size.saturating_sub(offset));
    if offset > buffer_size || buffer_size - offset < size {
        return Err(WorkGraphError::BackingMemoryOverrun {
            offset,
            size,
            buffer_size,
        });
    }
    let min_size = pipeline.memory_requirements.min_size;
    if size < min_size {
        return Err(WorkGraphError::BackingMemoryTooSmall { size, min_size });
    }
    Ok(size)
}

/// Validates input records in CPU memory against the entrypoints of `pipeline`.
pub(super) fn validate_cpu_inputs(
    pipeline: &WorkGraphPipeline,
    inputs: &[wgt::WorkGraphNodeInput<Vec<u8>>],
) -> Result<(), WorkGraphError> {
    let entrypoints = &pipeline.entrypoints;
    for (input_index, input) in inputs.iter().enumerate() {
        let Some(entrypoint) = entrypoints.get(input.entrypoint_index as usize) else {
            return Err(WorkGraphError::InvalidEntrypoint {
                input: input_index,
                index: input.entrypoint_index,
                count: entrypoints.len(),
            });
        };
        if entrypoint.record_size != 0 && input.record_stride != 0 {
            if input.record_stride < u64::from(entrypoint.record_size) {
                return Err(WorkGraphError::RecordStrideTooSmall {
                    input: input_index,
                    stride: input.record_stride,
                    record_size: entrypoint.record_size,
                });
            }
            let alignment = entrypoint.record_stride_alignment();
            if !input.record_stride.is_multiple_of(alignment) {
                return Err(WorkGraphError::UnalignedRecordStride {
                    input: input_index,
                    stride: input.record_stride,
                    alignment,
                });
            }
        }
        let required = entrypoint.required_records_size(input.num_records, input.record_stride);
        let size = input.records.len() as u64;
        if size < required {
            return Err(WorkGraphError::RecordsTooSmall {
                input: input_index,
                size,
                required,
            });
        }
    }

    if let Some(max) = pipeline.max_input_records {
        let records = inputs
            .iter()
            .map(|input| u64::from(input.num_records))
            .sum::<u64>();
        if records > u64::from(max.records) || inputs.len() > max.node_inputs as usize {
            return Err(WorkGraphError::TooManyInputRecords {
                records,
                node_inputs: inputs.len(),
                max_records: max.records,
                max_node_inputs: max.node_inputs,
            });
        }
    }

    Ok(())
}
