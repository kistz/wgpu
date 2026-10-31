//! Reflection of the work graph nodes in SPIR-V that uses `SPV_AMDX_shader_enqueue`.
//!
//! Vulkan has no query for the input record of a node, so the record layout comes from the
//! SPIR-V. The name of a node is the name of its entry point, because SPIR-V has no other
//! node name. Its array index comes from the `ShaderIndexAMDX` execution mode.

use alloc::{format, string::String, vec::Vec};

use naga::FastHashMap;

const MAGIC: u32 = 0x0723_0203;

const CAPABILITY_SHADER_ENQUEUE: u32 = 5067;
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const STORAGE_CLASS_NODE_PAYLOAD: u32 = 5068;
const STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

mod op {
    pub const ENTRY_POINT: u32 = 15;
    pub const EXECUTION_MODE: u32 = 16;
    pub const CAPABILITY: u32 = 17;
    pub const TYPE_INT: u32 = 21;
    pub const TYPE_FLOAT: u32 = 22;
    pub const TYPE_VECTOR: u32 = 23;
    pub const TYPE_MATRIX: u32 = 24;
    pub const TYPE_ARRAY: u32 = 28;
    pub const TYPE_RUNTIME_ARRAY: u32 = 29;
    pub const TYPE_STRUCT: u32 = 30;
    pub const TYPE_POINTER: u32 = 32;
    pub const CONSTANT_TRUE: u32 = 41;
    pub const CONSTANT_FALSE: u32 = 42;
    pub const CONSTANT: u32 = 43;
    pub const SPEC_CONSTANT_TRUE: u32 = 48;
    pub const SPEC_CONSTANT_FALSE: u32 = 49;
    pub const SPEC_CONSTANT: u32 = 50;
    pub const FUNCTION: u32 = 54;
    pub const FUNCTION_END: u32 = 56;
    pub const FUNCTION_CALL: u32 = 57;
    pub const VARIABLE: u32 = 59;
    pub const LOAD: u32 = 61;
    pub const STORE: u32 = 62;
    pub const COPY_MEMORY: u32 = 63;
    pub const COPY_MEMORY_SIZED: u32 = 64;
    pub const ACCESS_CHAIN: u32 = 65;
    pub const IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
    pub const PTR_ACCESS_CHAIN: u32 = 67;
    pub const IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
    pub const DECORATE: u32 = 71;
    pub const MEMBER_DECORATE: u32 = 72;
    pub const COPY_OBJECT: u32 = 83;
    pub const EXECUTION_MODE_ID: u32 = 331;
    pub const DECORATE_ID: u32 = 332;
    pub const ALLOCATE_NODE_PAYLOADS: u32 = 5074;
    pub const TYPE_NODE_PAYLOAD_ARRAY: u32 = 5076;
    pub const FINISH_WRITING_NODE_PAYLOAD: u32 = 5078;
    pub const NODE_PAYLOAD_ARRAY_LENGTH: u32 = 5090;
    pub const CONSTANT_STRING: u32 = 5103;
    pub const SPEC_CONSTANT_STRING: u32 = 5104;
}

mod decoration {
    pub const ROW_MAJOR: u32 = 4;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const MATRIX_STRIDE: u32 = 7;
    pub const OFFSET: u32 = 35;
    pub const PAYLOAD_NODE_NAME: u32 = 5091;
    pub const PAYLOAD_NODE_BASE_INDEX: u32 = 5098;
    pub const PAYLOAD_NODE_SPARSE_ARRAY: u32 = 5099;
    pub const PAYLOAD_NODE_ARRAY_SIZE: u32 = 5100;
}

mod mode {
    pub const COALESCING: u32 = 5069;
    pub const IS_API_ENTRY: u32 = 5070;
    pub const MAX_NODE_RECURSION: u32 = 5071;
    pub const STATIC_NUM_WORKGROUPS: u32 = 5072;
    pub const SHADER_INDEX: u32 = 5073;
    pub const MAX_NUM_WORKGROUPS: u32 = 5077;
    pub const SHARES_INPUT_WITH: u32 = 5102;
}

/// Size and alignment of the input record of a node, with C struct layout rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RecordLayout {
    pub size: u32,
    pub alignment: u32,
}

/// An output of a node, which sends records to the nodes with `name` and an array index in
/// `first_index..first_index + count`. A `count` of `None` is unbounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NodeOutput {
    pub name: String,
    pub first_index: u32,
    pub count: Option<u32>,
    pub sparse: bool,
}

impl NodeOutput {
    fn targets(&self, name: &str, index: u32) -> bool {
        self.name == name
            && index >= self.first_index
            && self.count.is_none_or(|count| {
                u64::from(index) < u64::from(self.first_index) + u64::from(count)
            })
    }
}

/// A compute shader entry point that is a work graph node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NodeShader {
    pub entry_point: String,
    pub index: u32,
    pub is_api_entry: Option<bool>,
    /// `None` if the node has no input record.
    pub input: Option<RecordLayout>,
    pub outputs: Vec<NodeOutput>,
}

/// Returns `true` if the SPIR-V declares the `ShaderEnqueueAMDX` capability.
pub(super) fn uses_shader_enqueue(spirv: &[u32]) -> bool {
    Instructions::new(spirv).is_ok_and(|instructions| {
        instructions
            .map_while(Result::ok)
            .take_while(|&(opcode, _)| opcode == op::CAPABILITY)
            .any(|(_, operands)| operands.first() == Some(&CAPABILITY_SHADER_ENQUEUE))
    })
}

/// Returns the work graph nodes of a SPIR-V module.
pub(super) fn reflect_nodes(spirv: &[u32]) -> Result<Vec<NodeShader>, String> {
    let module = Module::parse(spirv)?;
    let mut nodes = Vec::new();
    for entry_point in &module.entry_points {
        if let Some(node) = module.node(entry_point)? {
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// The nodes of a work graph, as indices into the nodes of the shader library.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Graph {
    pub nodes: Vec<usize>,
    pub entrypoints: Vec<usize>,
}

/// Selects the nodes and entrypoints of a work graph with the rules of D3D12 work graphs.
///
/// The graph contains all nodes of the library if `include_all_available_nodes` is set, and
/// otherwise `entrypoints` and the nodes they send records to. A node is an entrypoint if it is
/// listed in `entrypoints` or no other node sends records to it.
pub(super) fn select_graph(
    nodes: &[NodeShader],
    entrypoints: &[wgt::WorkGraphNodeId],
    include_all_available_nodes: bool,
) -> Result<Graph, String> {
    let find = |name: &str, index: u32| {
        nodes
            .iter()
            .position(|node| node.entry_point == name && node.index == index)
    };

    for (i, node) in nodes.iter().enumerate() {
        if find(&node.entry_point, node.index) != Some(i) {
            return Err(format!(
                "The shader library has more than one node {:?} with array index {}",
                node.entry_point, node.index
            ));
        }
    }

    let explicit_entrypoints = entrypoints
        .iter()
        .map(|id| {
            find(&id.name, id.array_index).ok_or_else(|| {
                format!(
                    "The shader library has no node {:?} with array index {}",
                    id.name, id.array_index
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut selected = if include_all_available_nodes {
        (0..nodes.len()).collect::<Vec<_>>()
    } else {
        let mut selected = Vec::new();
        let mut pending = explicit_entrypoints.clone();
        while let Some(i) = pending.pop() {
            if selected.contains(&i) {
                continue;
            }
            selected.push(i);
            for output in &nodes[i].outputs {
                pending.extend(
                    (0..nodes.len())
                        .filter(|&j| output.targets(&nodes[j].entry_point, nodes[j].index)),
                );
            }
        }
        selected.sort_unstable();
        selected
    };
    selected.dedup();

    for &i in &selected {
        for output in &nodes[i].outputs {
            let Some(count) = output.count.filter(|_| !output.sparse) else {
                continue;
            };
            let missing =
                (output.first_index..output.first_index.saturating_add(count)).find(|&index| {
                    !selected
                        .iter()
                        .any(|&j| nodes[j].entry_point == output.name && nodes[j].index == index)
                });
            if let Some(index) = missing {
                return Err(format!(
                    "Node {:?} sends records to node {:?} with array index {index}, which is \
                     not in the work graph. The name of a node is the name of its entry point",
                    nodes[i].entry_point, output.name
                ));
            }
        }
    }

    let entrypoints = selected
        .iter()
        .copied()
        .filter(|&i| {
            let node = &nodes[i];
            explicit_entrypoints.contains(&i)
                || node.is_api_entry.unwrap_or_else(|| {
                    !selected.iter().any(|&j| {
                        j != i
                            && nodes[j]
                                .outputs
                                .iter()
                                .any(|output| output.targets(&node.entry_point, node.index))
                    })
                })
        })
        .collect();

    Ok(Graph {
        nodes: selected,
        entrypoints,
    })
}

struct Instructions<'a> {
    words: &'a [u32],
}

impl<'a> Instructions<'a> {
    fn new(spirv: &'a [u32]) -> Result<Self, String> {
        match spirv {
            [MAGIC, _, _, _, _, words @ ..] => Ok(Self { words }),
            _ => Err(String::from("The shader is not SPIR-V")),
        }
    }
}

impl<'a> Iterator for Instructions<'a> {
    type Item = Result<(u32, &'a [u32]), String>;

    fn next(&mut self) -> Option<Self::Item> {
        let &first = self.words.first()?;
        let count = (first >> 16) as usize;
        if count == 0 || count > self.words.len() {
            self.words = &[];
            return Some(Err(String::from("The SPIR-V has a truncated instruction")));
        }
        let (instruction, rest) = self.words.split_at(count);
        self.words = rest;
        Some(Ok((first & 0xffff, &instruction[1..])))
    }
}

/// Decodes a literal string, and returns it with the number of words it occupies.
fn literal_string(words: &[u32]) -> Result<(String, usize), String> {
    let mut bytes = Vec::new();
    for (i, word) in words.iter().enumerate() {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                let string = String::from_utf8(bytes)
                    .map_err(|_| String::from("The SPIR-V has a string that is not UTF-8"))?;
                return Ok((string, i + 1));
            }
            bytes.push(byte);
        }
    }
    Err(String::from("The SPIR-V has an unterminated string"))
}

/// Returns the operands from `start` on, which may be none.
fn operands_from(operands: &[u32], start: usize) -> &[u32] {
    operands.get(start..).unwrap_or(&[])
}

fn operand(operands: &[u32], index: usize) -> Result<u32, String> {
    operands
        .get(index)
        .copied()
        .ok_or_else(|| String::from("The SPIR-V has an instruction with missing operands"))
}

enum Type {
    Scalar { width: u32 },
    Vector { component: u32, count: u32 },
    Matrix { column: u32, count: u32 },
    Array { element: u32, length: u32 },
    RuntimeArray { element: u32 },
    Struct { members: Vec<u32> },
    Pointer { storage_class: u32, pointee: u32 },
    NodePayloadArray { payload: u32 },
}

struct EntryPoint {
    function: u32,
    name: String,
    interface: Vec<u32>,
}

#[derive(Default)]
struct Function {
    /// `NodePayloadAMDX` variables that the function uses.
    payload_variables: Vec<u32>,
    /// Result types of `OpAllocateNodePayloadsAMDX`.
    allocations: Vec<u32>,
    calls: Vec<u32>,
}

type Decorations = Vec<(u32, Vec<u32>)>;

#[derive(Default)]
struct Module {
    types: FastHashMap<u32, Type>,
    constants: FastHashMap<u32, u64>,
    strings: FastHashMap<u32, String>,
    decorations: FastHashMap<u32, Decorations>,
    member_decorations: FastHashMap<(u32, u32), Decorations>,
    /// Pointee types of `NodePayloadAMDX` variables.
    payload_variables: FastHashMap<u32, u32>,
    entry_points: Vec<EntryPoint>,
    execution_modes: FastHashMap<u32, Decorations>,
    functions: FastHashMap<u32, Function>,
}

impl Module {
    fn parse(spirv: &[u32]) -> Result<Self, String> {
        let mut module = Self::default();
        let mut function = None;
        for instruction in Instructions::new(spirv)? {
            let (opcode, operands) = instruction?;
            let operand = |index| operand(operands, index);
            match opcode {
                op::ENTRY_POINT => {
                    if operand(0)? == EXECUTION_MODEL_GL_COMPUTE {
                        let (name, length) = literal_string(operands_from(operands, 2))?;
                        module.entry_points.push(EntryPoint {
                            function: operand(1)?,
                            name,
                            interface: operands_from(operands, 2 + length).to_vec(),
                        });
                    }
                }
                op::EXECUTION_MODE | op::EXECUTION_MODE_ID => {
                    module
                        .execution_modes
                        .entry(operand(0)?)
                        .or_default()
                        .push((operand(1)?, operands_from(operands, 2).to_vec()));
                }
                op::DECORATE | op::DECORATE_ID => {
                    module
                        .decorations
                        .entry(operand(0)?)
                        .or_default()
                        .push((operand(1)?, operands_from(operands, 2).to_vec()));
                }
                op::MEMBER_DECORATE => {
                    module
                        .member_decorations
                        .entry((operand(0)?, operand(1)?))
                        .or_default()
                        .push((operand(2)?, operands_from(operands, 3).to_vec()));
                }
                op::TYPE_INT | op::TYPE_FLOAT => {
                    module
                        .types
                        .insert(operand(0)?, Type::Scalar { width: operand(1)? });
                }
                op::TYPE_VECTOR => {
                    module.types.insert(
                        operand(0)?,
                        Type::Vector {
                            component: operand(1)?,
                            count: operand(2)?,
                        },
                    );
                }
                op::TYPE_MATRIX => {
                    module.types.insert(
                        operand(0)?,
                        Type::Matrix {
                            column: operand(1)?,
                            count: operand(2)?,
                        },
                    );
                }
                op::TYPE_ARRAY => {
                    module.types.insert(
                        operand(0)?,
                        Type::Array {
                            element: operand(1)?,
                            length: operand(2)?,
                        },
                    );
                }
                op::TYPE_RUNTIME_ARRAY => {
                    module.types.insert(
                        operand(0)?,
                        Type::RuntimeArray {
                            element: operand(1)?,
                        },
                    );
                }
                op::TYPE_STRUCT => {
                    module.types.insert(
                        operand(0)?,
                        Type::Struct {
                            members: operands_from(operands, 1).to_vec(),
                        },
                    );
                }
                op::TYPE_POINTER => {
                    module.types.insert(
                        operand(0)?,
                        Type::Pointer {
                            storage_class: operand(1)?,
                            pointee: operand(2)?,
                        },
                    );
                }
                op::TYPE_NODE_PAYLOAD_ARRAY => {
                    module.types.insert(
                        operand(0)?,
                        Type::NodePayloadArray {
                            payload: operand(1)?,
                        },
                    );
                }
                op::CONSTANT | op::SPEC_CONSTANT => {
                    let high = operands.get(3).copied().unwrap_or(0);
                    module.constants.insert(
                        operand(1)?,
                        u64::from(operand(2)?) | (u64::from(high) << 32),
                    );
                }
                op::CONSTANT_TRUE | op::SPEC_CONSTANT_TRUE => {
                    module.constants.insert(operand(1)?, 1);
                }
                op::CONSTANT_FALSE | op::SPEC_CONSTANT_FALSE => {
                    module.constants.insert(operand(1)?, 0);
                }
                op::CONSTANT_STRING | op::SPEC_CONSTANT_STRING => {
                    let (string, _) = literal_string(operands_from(operands, 1))?;
                    module.strings.insert(operand(0)?, string);
                }
                op::VARIABLE if function.is_none() => {
                    if operand(2)? == STORAGE_CLASS_NODE_PAYLOAD {
                        if let Some(&Type::Pointer { pointee, .. }) = module.types.get(&operand(0)?)
                        {
                            module.payload_variables.insert(operand(1)?, pointee);
                        }
                    }
                }
                op::FUNCTION => {
                    function = Some(operand(1)?);
                    module.functions.entry(operand(1)?).or_default();
                }
                op::FUNCTION_END => function = None,
                _ => {
                    let Some(id) = function else {
                        continue;
                    };
                    let pointers = match opcode {
                        op::FUNCTION_CALL => {
                            let callee = operand(2)?;
                            module.function(id).calls.push(callee);
                            operands_from(operands, 3)
                        }
                        op::ALLOCATE_NODE_PAYLOADS => {
                            let result_type = operand(0)?;
                            module.function(id).allocations.push(result_type);
                            &[]
                        }
                        op::STORE => &operands[..operands.len().min(1)],
                        op::COPY_MEMORY | op::COPY_MEMORY_SIZED => {
                            &operands[..operands.len().min(2)]
                        }
                        op::LOAD
                        | op::ACCESS_CHAIN
                        | op::IN_BOUNDS_ACCESS_CHAIN
                        | op::PTR_ACCESS_CHAIN
                        | op::IN_BOUNDS_PTR_ACCESS_CHAIN
                        | op::COPY_OBJECT
                        | op::FINISH_WRITING_NODE_PAYLOAD
                        | op::NODE_PAYLOAD_ARRAY_LENGTH => operands.get(2..3).unwrap_or(&[]),
                        _ => &[],
                    };
                    for &pointer in pointers {
                        if module.payload_variables.contains_key(&pointer) {
                            module.function(id).payload_variables.push(pointer);
                        }
                    }
                }
            }
        }
        Ok(module)
    }

    fn function(&mut self, id: u32) -> &mut Function {
        self.functions.entry(id).or_default()
    }

    fn constant(&self, id: u32) -> Result<u64, String> {
        self.constants
            .get(&id)
            .copied()
            .ok_or_else(|| format!("The SPIR-V has no constant with id {id}"))
    }

    fn constant_u32(&self, id: u32) -> Result<u32, String> {
        u32::try_from(self.constant(id)?)
            .map_err(|_| format!("The SPIR-V constant with id {id} does not fit in 32 bits"))
    }

    fn decoration<'a>(decorations: Option<&'a Decorations>, kind: u32) -> Option<&'a [u32]> {
        decorations?
            .iter()
            .find(|&&(decoration, _)| decoration == kind)
            .map(|(_, operands)| operands.as_slice())
    }

    /// Returns the payload variables and allocations of a function and the functions it calls.
    fn function_tree(&self, function: u32) -> (Vec<u32>, Vec<u32>) {
        let mut visited = Vec::new();
        let mut pending = alloc::vec![function];
        let (mut payload_variables, mut allocations) = (Vec::new(), Vec::new());
        while let Some(id) = pending.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.push(id);
            if let Some(function) = self.functions.get(&id) {
                payload_variables.extend(&function.payload_variables);
                allocations.extend(&function.allocations);
                pending.extend(&function.calls);
            }
        }
        payload_variables.sort_unstable();
        payload_variables.dedup();
        allocations.sort_unstable();
        allocations.dedup();
        (payload_variables, allocations)
    }

    fn node(&self, entry_point: &EntryPoint) -> Result<Option<NodeShader>, String> {
        let modes = self
            .execution_modes
            .get(&entry_point.function)
            .map_or(&[][..], Vec::as_slice);
        let mode = |kind| {
            modes
                .iter()
                .find(|&&(mode, _)| mode == kind)
                .map(|(_, operands)| operands.as_slice())
        };
        let (payload_variables, allocations) = self.function_tree(entry_point.function);

        let has_node_mode = modes.iter().any(|&(mode, _)| {
            matches!(
                mode,
                mode::COALESCING
                    | mode::IS_API_ENTRY
                    | mode::MAX_NODE_RECURSION
                    | mode::STATIC_NUM_WORKGROUPS
                    | mode::SHADER_INDEX
                    | mode::MAX_NUM_WORKGROUPS
                    | mode::SHARES_INPUT_WITH
            )
        });
        if !has_node_mode && payload_variables.is_empty() && allocations.is_empty() {
            return Ok(None);
        }
        let name = &entry_point.name;

        let index = match mode(mode::SHADER_INDEX) {
            Some(operands) => self.constant_u32(operand(operands, 0)?)?,
            None => 0,
        };
        let is_api_entry = match mode(mode::IS_API_ENTRY) {
            Some(operands) => Some(self.constant(operand(operands, 0)?)? != 0),
            None => None,
        };

        let input_variable = match *payload_variables.as_slice() {
            [] => self.unused_input_variable(entry_point),
            [variable] => Some(variable),
            _ => {
                return Err(format!(
                    "Node {name:?} uses more than one input record variable"
                ))
            }
        };
        let input = input_variable
            .map(|variable| {
                let payload = self.record_type(self.payload_variables[&variable]);
                self.layout(payload, None)
                    .map_err(|error| format!("The input record of node {name:?}: {error}"))
            })
            .transpose()?
            .map(|(size, alignment)| RecordLayout { size, alignment });

        let mut outputs = allocations
            .iter()
            .map(|&ty| self.output(ty))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("An output of node {name:?}: {error}"))?;
        outputs.dedup();

        Ok(Some(NodeShader {
            entry_point: name.clone(),
            index,
            is_api_entry,
            input,
            outputs,
        }))
    }

    /// Returns the input record variable of a node whose code does not read its input record.
    ///
    /// DXC lists every payload variable of the module in the interface of every entry point, so
    /// a variable in the interface is only the node's input if no entry point uses it.
    fn unused_input_variable(&self, entry_point: &EntryPoint) -> Option<u32> {
        let used = self
            .entry_points
            .iter()
            .flat_map(|entry_point| self.function_tree(entry_point.function).0)
            .collect::<Vec<_>>();
        let mut candidates = entry_point
            .interface
            .iter()
            .copied()
            .filter(|id| self.payload_variables.contains_key(id) && !used.contains(id));
        match (candidates.next(), candidates.next()) {
            (Some(variable), None) => Some(variable),
            _ => None,
        }
    }

    /// Returns the record type of a payload variable's pointee type.
    fn record_type(&self, ty: u32) -> u32 {
        match self.types.get(&ty) {
            Some(
                &Type::NodePayloadArray { payload: element }
                | &Type::RuntimeArray { element }
                | &Type::Array { element, .. },
            ) => element,
            _ => ty,
        }
    }

    fn output(&self, allocation_type: u32) -> Result<NodeOutput, String> {
        let ty = match self.types.get(&allocation_type) {
            Some(&Type::Pointer { pointee, .. }) => pointee,
            _ => allocation_type,
        };
        let decorations = self.decorations.get(&ty);
        let name = Self::decoration(decorations, decoration::PAYLOAD_NODE_NAME)
            .and_then(|operands| self.strings.get(operands.first()?))
            .ok_or_else(|| String::from("The output has no node name"))?;
        let first_index = match Self::decoration(decorations, decoration::PAYLOAD_NODE_BASE_INDEX) {
            Some(operands) => self.constant_u32(operand(operands, 0)?)?,
            None => 0,
        };
        let sparse = Self::decoration(decorations, decoration::PAYLOAD_NODE_SPARSE_ARRAY).is_some();
        let count = match Self::decoration(decorations, decoration::PAYLOAD_NODE_ARRAY_SIZE) {
            Some(operands) => {
                Some(self.constant_u32(operand(operands, 0)?)?).filter(|&count| count != u32::MAX)
            }
            None if sparse => None,
            None => Some(1),
        };
        Ok(NodeOutput {
            name: name.clone(),
            first_index,
            count,
            sparse,
        })
    }

    /// Returns the size and alignment of a type with C struct layout rules.
    ///
    /// `member` is the struct and member index of the type, which matrices need for their stride.
    fn layout(&self, ty: u32, member: Option<(u32, u32)>) -> Result<(u32, u32), String> {
        let member_decorations = member.and_then(|member| self.member_decorations.get(&member));
        let (size, alignment) = match self.types.get(&ty) {
            Some(&Type::Scalar { width }) => {
                let size = u64::from(width / 8);
                (size, size)
            }
            Some(&Type::Vector { component, count }) => {
                let (size, alignment) = self.layout(component, None)?;
                (u64::from(size) * u64::from(count), u64::from(alignment))
            }
            Some(&Type::Matrix { column, count }) => {
                let Some(&Type::Vector {
                    component,
                    count: rows,
                }) = self.types.get(&column)
                else {
                    return Err(String::from("A matrix has columns that are not vectors"));
                };
                let stride = Self::decoration(member_decorations, decoration::MATRIX_STRIDE)
                    .and_then(|operands| operands.first())
                    .ok_or_else(|| String::from("A matrix has no matrix stride"))?;
                let vectors =
                    if Self::decoration(member_decorations, decoration::ROW_MAJOR).is_some() {
                        rows
                    } else {
                        count
                    };
                let (_, alignment) = self.layout(component, None)?;
                (
                    u64::from(*stride) * u64::from(vectors),
                    u64::from(alignment),
                )
            }
            Some(&Type::Array { element, length }) => {
                let (size, alignment) = self.layout(element, None)?;
                let stride =
                    match Self::decoration(self.decorations.get(&ty), decoration::ARRAY_STRIDE) {
                        Some(operands) => operand(operands, 0)?,
                        None => size,
                    };
                (
                    u64::from(stride) * self.constant(length)?,
                    u64::from(alignment),
                )
            }
            Some(Type::Struct { members }) => {
                let (mut size, mut alignment) = (0, 1);
                for (index, &member_type) in members.iter().enumerate() {
                    let index = index as u32;
                    let offset = Self::decoration(
                        self.member_decorations.get(&(ty, index)),
                        decoration::OFFSET,
                    )
                    .and_then(|operands| operands.first())
                    .ok_or_else(|| String::from("A struct member has no offset"))?;
                    let (member_size, member_alignment) =
                        self.layout(member_type, Some((ty, index)))?;
                    size = size.max(u64::from(*offset) + u64::from(member_size));
                    alignment = alignment.max(u64::from(member_alignment));
                }
                (size.next_multiple_of(alignment), alignment)
            }
            Some(&Type::Pointer { storage_class, .. })
                if storage_class == STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER =>
            {
                (8, 8)
            }
            _ => return Err(format!("The type with id {ty} is not supported in records")),
        };
        let size =
            u32::try_from(size).map_err(|_| String::from("The record is larger than 4 GiB"))?;
        Ok((size, alignment.max(1) as u32))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{borrow::Cow, string::ToString, vec};

    use super::*;

    /// Builds a SPIR-V module from instructions.
    #[derive(Default)]
    struct Assembler {
        words: Vec<u32>,
    }

    impl Assembler {
        fn op(mut self, opcode: u32, operands: &[u32]) -> Self {
            self.words
                .push(((operands.len() as u32 + 1) << 16) | opcode);
            self.words.extend_from_slice(operands);
            self
        }

        fn finish(self) -> Vec<u32> {
            let mut words = vec![MAGIC, 0x0001_0600, 0, 1000, 0];
            words.extend(self.words);
            words
        }
    }

    fn string(string: &str) -> Vec<u32> {
        let mut bytes = string.as_bytes().to_vec();
        bytes.resize((bytes.len() / 4 + 1) * 4, 0);
        bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn with_string(prefix: &[u32], string_operand: &str, suffix: &[u32]) -> Vec<u32> {
        let mut operands = prefix.to_vec();
        operands.extend(string(string_operand));
        operands.extend_from_slice(suffix);
        operands
    }

    const UINT: u32 = 1;
    const UINT3: u32 = 2;
    const ZERO: u32 = 3;
    const ONE: u32 = 4;
    const FOUR: u32 = 5;
    const ENTRY_RECORD: u32 = 10;
    const ENTRY_PAYLOADS: u32 = 11;
    const ENTRY_POINTER: u32 = 12;
    const ENTRY_INPUT: u32 = 13;
    const LEAF_RECORD: u32 = 20;
    const LEAF_PAYLOADS: u32 = 21;
    const LEAF_POINTER: u32 = 22;
    const LEAF_INPUT: u32 = 23;
    const LEAF_OUTPUT_PAYLOADS: u32 = 24;
    const LEAF_OUTPUT_POINTER: u32 = 25;
    const LEAF_NAME: u32 = 26;
    const ENTRY: u32 = 30;
    const LEAF: u32 = 31;
    const UINT_POINTER: u32 = 40;

    /// The library of the work graph GPU tests, as DXC compiles it: a broadcasting `entry` node
    /// whose record has a dispatch grid and a value, and which sends records to a `leaf` node.
    fn library(leaf_name: &str) -> Vec<u32> {
        Assembler::default()
            .op(op::CAPABILITY, &[1])
            .op(op::CAPABILITY, &[CAPABILITY_SHADER_ENQUEUE])
            .op(
                op::ENTRY_POINT,
                &with_string(
                    &[EXECUTION_MODEL_GL_COMPUTE, ENTRY],
                    "entry",
                    &[ENTRY_INPUT, LEAF_INPUT],
                ),
            )
            .op(
                op::ENTRY_POINT,
                &with_string(
                    &[EXECUTION_MODEL_GL_COMPUTE, LEAF],
                    leaf_name,
                    &[ENTRY_INPUT, LEAF_INPUT],
                ),
            )
            .op(op::EXECUTION_MODE_ID, &[ENTRY, mode::SHADER_INDEX, ZERO])
            .op(
                op::EXECUTION_MODE_ID,
                &[ENTRY, mode::MAX_NUM_WORKGROUPS, FOUR, ONE, ONE],
            )
            .op(op::EXECUTION_MODE, &[LEAF, mode::COALESCING])
            .op(op::EXECUTION_MODE_ID, &[LEAF, mode::SHADER_INDEX, ZERO])
            .op(
                op::MEMBER_DECORATE,
                &[ENTRY_RECORD, 0, decoration::OFFSET, 0],
            )
            .op(
                op::MEMBER_DECORATE,
                &[ENTRY_RECORD, 1, decoration::OFFSET, 12],
            )
            .op(
                op::MEMBER_DECORATE,
                &[LEAF_RECORD, 0, decoration::OFFSET, 0],
            )
            .op(
                op::MEMBER_DECORATE,
                &[LEAF_RECORD, 1, decoration::OFFSET, 4],
            )
            .op(
                op::DECORATE_ID,
                &[
                    LEAF_OUTPUT_PAYLOADS,
                    decoration::PAYLOAD_NODE_NAME,
                    LEAF_NAME,
                ],
            )
            .op(op::TYPE_INT, &[UINT, 32, 0])
            .op(op::CONSTANT, &[UINT, ZERO, 0])
            .op(op::CONSTANT, &[UINT, ONE, 1])
            .op(op::CONSTANT, &[UINT, FOUR, 4])
            .op(op::TYPE_VECTOR, &[UINT3, UINT, 3])
            .op(op::TYPE_STRUCT, &[ENTRY_RECORD, UINT3, UINT])
            .op(op::TYPE_NODE_PAYLOAD_ARRAY, &[ENTRY_PAYLOADS, ENTRY_RECORD])
            .op(
                op::TYPE_POINTER,
                &[ENTRY_POINTER, STORAGE_CLASS_NODE_PAYLOAD, ENTRY_PAYLOADS],
            )
            .op(op::TYPE_STRUCT, &[LEAF_RECORD, UINT, UINT])
            .op(op::TYPE_NODE_PAYLOAD_ARRAY, &[LEAF_PAYLOADS, LEAF_RECORD])
            .op(
                op::TYPE_POINTER,
                &[LEAF_POINTER, STORAGE_CLASS_NODE_PAYLOAD, LEAF_PAYLOADS],
            )
            .op(op::CONSTANT_STRING, &with_string(&[LEAF_NAME], "leaf", &[]))
            .op(
                op::TYPE_NODE_PAYLOAD_ARRAY,
                &[LEAF_OUTPUT_PAYLOADS, LEAF_RECORD],
            )
            .op(
                op::TYPE_POINTER,
                &[
                    LEAF_OUTPUT_POINTER,
                    STORAGE_CLASS_NODE_PAYLOAD,
                    LEAF_OUTPUT_PAYLOADS,
                ],
            )
            .op(
                op::TYPE_POINTER,
                &[UINT_POINTER, STORAGE_CLASS_NODE_PAYLOAD, UINT],
            )
            .op(
                op::VARIABLE,
                &[ENTRY_POINTER, ENTRY_INPUT, STORAGE_CLASS_NODE_PAYLOAD],
            )
            .op(
                op::VARIABLE,
                &[LEAF_POINTER, LEAF_INPUT, STORAGE_CLASS_NODE_PAYLOAD],
            )
            .op(op::FUNCTION, &[0, ENTRY, 0, 0])
            .op(
                op::ALLOCATE_NODE_PAYLOADS,
                &[LEAF_OUTPUT_POINTER, 50, FOUR, ONE, ZERO],
            )
            .op(
                op::ACCESS_CHAIN,
                &[UINT_POINTER, 51, ENTRY_INPUT, ZERO, ONE],
            )
            .op(op::FUNCTION_END, &[])
            .op(op::FUNCTION, &[0, LEAF, 0, 0])
            .op(
                op::ACCESS_CHAIN,
                &[UINT_POINTER, 52, LEAF_INPUT, ZERO, ZERO],
            )
            .op(op::FUNCTION_END, &[])
            .finish()
    }

    fn node_id(name: &str, array_index: u32) -> wgt::WorkGraphNodeId<'_> {
        wgt::WorkGraphNodeId {
            name: Cow::Borrowed(name),
            array_index,
        }
    }

    #[test]
    fn reflects_nodes() {
        let spirv = library("leaf");
        assert!(uses_shader_enqueue(&spirv));
        assert_eq!(
            reflect_nodes(&spirv).unwrap(),
            [
                NodeShader {
                    entry_point: "entry".to_string(),
                    index: 0,
                    is_api_entry: None,
                    input: Some(RecordLayout {
                        size: 16,
                        alignment: 4
                    }),
                    outputs: vec![NodeOutput {
                        name: "leaf".to_string(),
                        first_index: 0,
                        count: Some(1),
                        sparse: false,
                    }],
                },
                NodeShader {
                    entry_point: "leaf".to_string(),
                    index: 0,
                    is_api_entry: None,
                    input: Some(RecordLayout {
                        size: 8,
                        alignment: 4
                    }),
                    outputs: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn selects_entrypoints() {
        let nodes = reflect_nodes(&library("leaf")).unwrap();
        let graph = Graph {
            nodes: vec![0, 1],
            entrypoints: vec![0],
        };
        assert_eq!(select_graph(&nodes, &[], true).unwrap(), graph);
        assert_eq!(
            select_graph(&nodes, &[node_id("entry", 0)], false).unwrap(),
            graph
        );
        assert_eq!(
            select_graph(&nodes, &[node_id("leaf", 0)], false).unwrap(),
            Graph {
                nodes: vec![1],
                entrypoints: vec![1],
            }
        );
        assert_eq!(
            select_graph(&nodes, &[node_id("leaf", 0)], true).unwrap(),
            Graph {
                nodes: vec![0, 1],
                entrypoints: vec![0, 1],
            }
        );
    }

    #[test]
    fn rejects_missing_nodes() {
        let nodes = reflect_nodes(&library("renamed_leaf")).unwrap();
        let error = select_graph(&nodes, &[], true).unwrap_err();
        assert!(error.contains("\"leaf\""), "{error}");
        let error = select_graph(&nodes, &[node_id("entry", 1)], false).unwrap_err();
        assert!(
            error.contains("no node \"entry\" with array index 1"),
            "{error}"
        );
    }

    #[test]
    fn rejects_non_spirv() {
        assert!(!uses_shader_enqueue(&[1, 2, 3]));
        assert!(reflect_nodes(&[1, 2, 3]).is_err());
    }
}
