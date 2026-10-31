struct EntryRecord
{
    float4 color;
};

struct Vertex
{
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[Shader("node")]
[NodeIsProgramEntry]
[NodeLaunch("thread")]
void entry(
    ThreadNodeInputRecord<EntryRecord> input,
    [MaxRecords(1)] [NodeId("triangle")] NodeOutput<EntryRecord> output)
{
    ThreadNodeOutputRecords<EntryRecord> record = output.GetThreadNodeOutputRecords(1);
    record.Get().color = input.Get().color;
    record.OutputComplete();
}

// Draws a triangle that covers the render target.
[Shader("node")]
[NodeLaunch("mesh")]
[NodeDispatchGrid(1, 1, 1)]
[NumThreads(3, 1, 1)]
[OutputTopology("triangle")]
void triangle_mesh(
    DispatchNodeInputRecord<EntryRecord> input,
    uint thread : SV_GroupThreadID,
    out vertices Vertex vertices[3],
    out indices uint3 triangles[1])
{
    SetMeshOutputCounts(3, 1);
    const float2 positions[3] = { float2(-1, -1), float2(-1, 3), float2(3, -1) };
    vertices[thread].position = float4(positions[thread], 0, 1);
    vertices[thread].color = input.Get().color;
    if (thread == 0)
    {
        triangles[0] = uint3(0, 1, 2);
    }
}

[Shader("pixel")]
float4 triangle_fragment(Vertex input) : SV_Target
{
    return input.color;
}
