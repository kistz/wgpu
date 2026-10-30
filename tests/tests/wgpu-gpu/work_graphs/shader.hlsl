RWByteAddressBuffer output : register(u0);

struct EntryRecord {
    uint3 grid : SV_DispatchGrid;
    uint value;
};

struct LeafRecord {
    uint index;
    uint value;
};

[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeMaxDispatchGrid(4, 1, 1)]
[NumThreads(4, 1, 1)]
void entry(
    DispatchNodeInputRecord<EntryRecord> input,
    uint thread_id : SV_DispatchThreadID,
    [MaxRecords(4)] NodeOutput<LeafRecord> leaf)
{
    ThreadNodeOutputRecords<LeafRecord> record = leaf.GetThreadNodeOutputRecords(1);
    record.Get().index = thread_id;
    record.Get().value = input.Get().value;
    record.OutputComplete();
}

[Shader("node")]
[NodeLaunch("thread")]
void leaf(ThreadNodeInputRecord<LeafRecord> input)
{
    uint original;
    output.InterlockedAdd(input.Get().index * 4, input.Get().value, original);
}
