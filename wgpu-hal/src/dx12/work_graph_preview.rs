//! Work graph declarations that only preview Agility SDKs define.
//!
//! The layouts match `d3d12.h` of Agility SDK 1.722.0-preview.

use core::{ffi::c_void, ptr};

use windows::{
    core::{IUnknown, Interface, BOOL, GUID, PCWSTR},
    Win32::Graphics::Direct3D12,
};

pub const D3D12_WORK_GRAPHS_TIER_1_1: Direct3D12::D3D12_WORK_GRAPHS_TIER =
    Direct3D12::D3D12_WORK_GRAPHS_TIER(11);

pub const D3D12_STATE_OBJECT_FLAG_WORK_GRAPHS_USE_GRAPHICS_STATE_FOR_GLOBAL_ROOT_SIGNATURE:
    Direct3D12::D3D12_STATE_OBJECT_FLAGS = Direct3D12::D3D12_STATE_OBJECT_FLAGS(0x40);

pub const D3D12_NODE_TYPE_PROGRAM: Direct3D12::D3D12_NODE_TYPE = Direct3D12::D3D12_NODE_TYPE(1);

pub const D3D12_PROGRAM_NODE_OVERRIDES_TYPE_NONE: i32 = 0;
pub const D3D12_PROGRAM_NODE_OVERRIDES_TYPE_MESH_LAUNCH: i32 = 3;

/// `D3D12_MESH_LAUNCH_OVERRIDES`
#[repr(C)]
pub struct MeshLaunchOverrides {
    pub local_root_arguments_table_index: *const u32,
    pub program_entry: *const BOOL,
    pub new_name: *const Direct3D12::D3D12_NODE_ID,
    pub share_input_of: *const Direct3D12::D3D12_NODE_ID,
    pub dispatch_grid: *const u32,
    pub max_dispatch_grid: *const u32,
    pub max_input_records_per_graph_entry_record: *const c_void,
}

/// `D3D12_NODE` with `NodeType` set to [`D3D12_NODE_TYPE_PROGRAM`], which holds a
/// `D3D12_PROGRAM_NODE`.
#[repr(C)]
pub struct ProgramNode {
    pub node_type: Direct3D12::D3D12_NODE_TYPE,
    pub program: PCWSTR,
    pub overrides_type: i32,
    pub overrides: *const MeshLaunchOverrides,
}

const _: () = assert!(size_of::<ProgramNode>() == size_of::<Direct3D12::D3D12_NODE>());

/// Calls `ID3D12WorkGraphProperties1::SetMaximumInputRecords`.
pub fn set_maximum_input_records(
    properties: &Direct3D12::ID3D12WorkGraphProperties,
    work_graph_index: u32,
    max_records: u32,
    max_node_inputs: u32,
) -> windows::core::Result<()> {
    const IID_ID3D12_WORK_GRAPH_PROPERTIES1: GUID =
        GUID::from_u128(0x5490ef66_165f_4b3f_9658_74e5c6d2e1d0);

    #[repr(C)]
    struct Vtable {
        base: Direct3D12::ID3D12WorkGraphProperties_Vtbl,
        set_maximum_input_records: unsafe extern "system" fn(*mut c_void, u32, u32, u32),
    }

    let mut raw = ptr::null_mut();
    unsafe { properties.query(&IID_ID3D12_WORK_GRAPH_PROPERTIES1, &mut raw) }.ok()?;
    // Releases the reference that `query` added.
    let properties1 = unsafe { IUnknown::from_raw(raw) };
    unsafe {
        let vtable = *raw.cast::<*const Vtable>();
        ((*vtable).set_maximum_input_records)(
            properties1.as_raw(),
            work_graph_index,
            max_records,
            max_node_inputs,
        );
    }
    Ok(())
}
