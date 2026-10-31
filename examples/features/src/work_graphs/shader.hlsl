// A work graph that draws a Koch snowflake and a Sierpinski carpet into `Canvas`.
// Lines and squares are split further the closer they are to `Focus`.

[[vk::binding(0)]] [[vk::image_format("rgba8")]]
RWTexture2D<unorm float4> Canvas : register(u0);

[[vk::binding(1)]]
cbuffer Scene : register(b0)
{
    float2 Focus;
    float FocusRadius;
};

#define FILL_GROUP_SIZE 8
#define MAX_FILL_GRID 32

static const float PI = 3.14159265;
static const uint SNOWFLAKE = 0;

struct Fractal
{
    float2 center;
    float radius;
    float rotation;
    uint kind;
};

struct Line
{
    float2 a;
    float2 b;
};

struct Square
{
    float2 center;
    float halfSize;
    float rotation;
};

struct FillRecord
{
    uint2 grid : SV_DispatchGrid;
    float2 center;
    float halfSize;
    float rotation;
};

// 1 at `Focus`, falling to 0 at `FocusRadius`.
float Detail(float2 position)
{
    return 1 - smoothstep(0, FocusRadius, distance(position, Focus));
}

// Lines and squares are split until they are smaller than this, in pixels.
float MinSize(float2 position)
{
    return lerp(96, 5, Detail(position));
}

float4 Color(float2 position)
{
    const float t = 0.6 - 0.45 * Detail(position);
    return float4(0.5 + 0.5 * cos(2 * PI * (t + float3(0.0, 0.33, 0.67))), 1);
}

float2x2 Rotation(float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return float2x2(c, -s, s, c);
}

float2 TriangleCorner(Fractal fractal, uint index)
{
    const float angle = fractal.rotation - index * 2 * PI / 3;
    return fractal.center + fractal.radius * float2(sin(angle), -cos(angle));
}

[Shader("node")]
[NodeIsProgramEntry]
[NodeLaunch("thread")]
void Entry(
    ThreadNodeInputRecord<Fractal> input,
    [MaxRecords(3)] [NodeId("Snowflake")] NodeOutput<Line> snowflakeOutput,
    [MaxRecords(1)] [NodeId("Carpet")] NodeOutput<Square> carpetOutput)
{
    const Fractal fractal = input.Get();
    const bool isSnowflake = fractal.kind == SNOWFLAKE;

    // Records must be requested and completed in thread group uniform control flow, so unused
    // outputs request zero records.
    ThreadNodeOutputRecords<Line> lines =
        snowflakeOutput.GetThreadNodeOutputRecords(isSnowflake ? 3 : 0);
    ThreadNodeOutputRecords<Square> squares =
        carpetOutput.GetThreadNodeOutputRecords(isSnowflake ? 0 : 1);

    if (isSnowflake)
    {
        for (uint i = 0; i < 3; ++i)
        {
            lines.Get(i).a = TriangleCorner(fractal, i);
            lines.Get(i).b = TriangleCorner(fractal, i + 1);
        }
    }
    else
    {
        squares.Get().center = fractal.center;
        squares.Get().halfSize = fractal.radius * 0.75;
        squares.Get().rotation = fractal.rotation;
    }

    lines.OutputComplete();
    squares.OutputComplete();
}

[Shader("node")]
[NodeLaunch("thread")]
[NodeMaxRecursionDepth(5)]
void Snowflake(
    ThreadNodeInputRecord<Line> input,
    [MaxRecords(4)] [NodeId("Snowflake")] NodeOutput<Line> splitOutput,
    [MaxRecords(1)] [NodeId("DrawLine")] NodeOutput<Line> drawOutput)
{
    const float2 a = input.Get().a;
    const float2 b = input.Get().b;
    const bool split = GetRemainingRecursionLevels() != 0 && distance(a, b) > MinSize((a + b) / 2);

    ThreadNodeOutputRecords<Line> lines = splitOutput.GetThreadNodeOutputRecords(split ? 4 : 0);
    ThreadNodeOutputRecords<Line> drawnLine = drawOutput.GetThreadNodeOutputRecords(split ? 0 : 1);

    if (split)
    {
        // Replaces the middle third with the other two sides of an equilateral triangle.
        const float2 normal = float2(a.y - b.y, b.x - a.x) * sqrt(3) / 6;
        const float2 points[5] = { a, lerp(a, b, 1.0 / 3), (a + b) / 2 + normal, lerp(a, b, 2.0 / 3), b };
        for (uint i = 0; i < 4; ++i)
        {
            lines.Get(i).a = points[i];
            lines.Get(i).b = points[i + 1];
        }
    }
    else
    {
        drawnLine.Get().a = a;
        drawnLine.Get().b = b;
    }

    lines.OutputComplete();
    drawnLine.OutputComplete();
}

// Draws up to 32 lines per thread group, one per thread.
[Shader("node")]
[NodeLaunch("coalescing")]
[NumThreads(32, 1, 1)]
void DrawLine(
    [MaxRecords(32)] GroupNodeInputRecords<Line> lines,
    uint thread : SV_GroupIndex)
{
    if (thread >= lines.Count())
    {
        return;
    }

    const float2 a = lines[thread].a;
    const float2 b = lines[thread].b;
    const uint steps = max(uint(ceil(max(abs(b.x - a.x), abs(b.y - a.y)))), 1);
    for (uint i = 0; i <= steps; ++i)
    {
        const float2 position = lerp(a, b, float(i) / steps);
        const float4 color = Color(position);
        const int2 pixel = int2(floor(position));
        Canvas[pixel] = color;
        Canvas[pixel + int2(1, 0)] = color;
        Canvas[pixel + int2(0, 1)] = color;
        Canvas[pixel + int2(1, 1)] = color;
    }
}

// Splits a square into 3x3 squares and leaves out the middle one. Each of the 8 threads either
// splits its square further or fills it.
[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1, 1, 1)]
[NumThreads(8, 1, 1)]
[NodeMaxRecursionDepth(4)]
void Carpet(
    DispatchNodeInputRecord<Square> input,
    uint thread : SV_GroupIndex,
    [MaxRecords(8)] [NodeId("Carpet")] NodeOutput<Square> splitOutput,
    [MaxRecords(8)] [NodeId("FillSquare")] NodeOutput<FillRecord> fillOutput)
{
    const Square parent = input.Get();
    const uint cell = thread < 4 ? thread : thread + 1;
    const float2 offset = float2(int(cell % 3) - 1, int(cell / 3) - 1) * (2 * parent.halfSize / 3);
    const float2 center = parent.center + mul(Rotation(parent.rotation), offset);
    const float halfSize = parent.halfSize / 3;
    const bool split = GetRemainingRecursionLevels() != 0 && 2 * halfSize > MinSize(center);

    ThreadNodeOutputRecords<Square> square = splitOutput.GetThreadNodeOutputRecords(split ? 1 : 0);
    ThreadNodeOutputRecords<FillRecord> fill = fillOutput.GetThreadNodeOutputRecords(split ? 0 : 1);

    if (split)
    {
        square.Get().center = center;
        square.Get().halfSize = halfSize;
        square.Get().rotation = parent.rotation;
    }
    else
    {
        // Covers the bounding box of the rotated square with thread groups.
        const float extent = halfSize * (abs(cos(parent.rotation)) + abs(sin(parent.rotation)));
        const uint groups = min(uint(ceil((2 * extent + 2) / FILL_GROUP_SIZE)), MAX_FILL_GRID);
        fill.Get().grid = uint2(groups, groups);
        fill.Get().center = center;
        fill.Get().halfSize = halfSize;
        fill.Get().rotation = parent.rotation;
    }

    square.OutputComplete();
    fill.OutputComplete();
}

// Fills a square with one thread per pixel. The size of the dispatch grid depends on the square.
[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeMaxDispatchGrid(MAX_FILL_GRID, MAX_FILL_GRID, 1)]
[NumThreads(FILL_GROUP_SIZE, FILL_GROUP_SIZE, 1)]
void FillSquare(
    DispatchNodeInputRecord<FillRecord> input,
    uint2 thread : SV_DispatchThreadID)
{
    const FillRecord square = input.Get();
    const int2 pixel = int2(floor(square.center)) - int2(square.grid * FILL_GROUP_SIZE / 2) + int2(thread);
    const float2 local = mul(float2(pixel) + 0.5 - square.center, Rotation(square.rotation));

    // Leaves a gap between neighboring squares.
    if (max(abs(local.x), abs(local.y)) < square.halfSize - 0.75)
    {
        Canvas[pixel] = Color(square.center);
    }
}
