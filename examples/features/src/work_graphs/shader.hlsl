// FILE BORROWED: All Rights remain to the original authors. https://github.com/GPUOpen-LibrariesAndSDKs/WorkGraphPlayground

// This file is part of the AMD & HSC Work Graph Playground.
//
// Copyright (C) 2025 Advanced Micro Devices, Inc. and Coburg University of Applied Sciences and Arts.
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "Common.h"

struct Line
{
    float2 a, b;
};

// [Task 2 Solution]:
struct Box
{
    float2 topLeft;
    float size;
};

[Shader("node")]
    [NodeIsProgramEntry]
    [NodeLaunch("thread")]
    [NodeId("Entry")] void EntryNode(
        [MaxRecords(3)]
            [NodeId("Snowflake")] NodeOutput<Line>
                snowflakeOutput,

        // [Task 2 Solution]: Single record output to the "Sponge" node.
        [MaxRecords(1)]
            [NodeId("Sponge")] NodeOutput<Box>
                spongeOutput)
{
    const bool stackVertical = RenderSize.x > RenderSize.y;
    const float scale = stackVertical ? min(RenderSize.x * .225, RenderSize.y * .45) : min(RenderSize.x * .45, RenderSize.y * .225);
    // Draw Snowflake fractal.
    {
        const float2 snowflakeCenter = RenderSize * (stackVertical ? float2(.25, .5) : float2(.5, .25));

        // Request three output records for the three sides of the initial equilateral triangle.
        ThreadNodeOutputRecords<Line> outputRecords =
            snowflakeOutput.GetThreadNodeOutputRecords(3);

        // Compute three vertices of the initial equilateral triangle.
        const float2 v0 = snowflakeCenter + scale * float2(0., -1.);
        const float2 v1 = snowflakeCenter + scale * float2(-sqrt(3) * .5, .5);
        const float2 v2 = snowflakeCenter + scale * float2(+sqrt(3) * .5, .5);

        // Create the initial equilateral triangle.
        outputRecords.Get(0).a = v0;
        outputRecords.Get(0).b = v1;

        outputRecords.Get(1).a = v1;
        outputRecords.Get(2).a = v2;

        outputRecords.Get(2).b = v0;
        outputRecords.Get(1).b = v2;

        outputRecords.OutputComplete();
    }

    // Draw Sponge fractal.
    {
        const float2 spongeCenter = RenderSize * (stackVertical ? float2(.75, .5) : float2(.5, .75));

        // [Solution 2]: Request a single record for the "Sponge" node and write the initial box
        //               position and size to it.
        ThreadNodeOutputRecords<Box> outputRecord = spongeOutput.GetThreadNodeOutputRecords(1);

        outputRecord.Get().topLeft = spongeCenter - scale;
        outputRecord.Get().size = 2 * scale;

        outputRecord.OutputComplete();
    }
}

    [Shader("node")]
    [NodeLaunch("thread")]
    // If a node declares a recursive output to itself (see "recursiveOutput" below),
    // a "[NodeMaxRecursionDepth(...)]" is required to specify the maximum number of recursion levels.
    // This is required, as each recursion level counts towards the total graph depth,
    // and the runtime has to ensure that this depth does not exceed the limit of 32 nodes.
    // We can use "GetRemainingRecursionLevels()" in the node function to query the remaining
    // recursion levels, i.e., determine whether we can still request recursive output records.
    [NodeMaxRecursionDepth(4)]
    [NodeId("Snowflake")] void SnowflakeNode(
        ThreadNodeInputRecord<Line> inputRecord,

        [MaxRecords(4)]
            [NodeId("Snowflake")] NodeOutput<Line>
                recursiveOutput)
{
    const float2 a = inputRecord.Get().a;
    const float2 b = inputRecord.Get().b;

    // Check if we have reached the recursion limit.
    const bool hasOutput = GetRemainingRecursionLevels() != 0;

    // Each recursion level has a 4x amplification factor, as each line
    // splits into four new lines.
    ThreadNodeOutputRecords<Line> outputRecords =
        recursiveOutput.GetThreadNodeOutputRecords(hasOutput * 4);

    if (hasOutput)
    {
        // Perpendicular vector to current line segment.
        const float2 perp = float2(a.y - b.y, b.x - a.x) * sqrt(3) / 6;

        // Compute vertices for the four new line segments:
        //
        //             v2
        //            /  \
        //           /    \
        // v0 ---- v1      v3 ---- v4
        const float2 v0 = a;
        const float2 v1 = lerp(a, b, 1. / 3.);
        const float2 v2 = lerp(a, b, .5) + perp;
        const float2 v3 = lerp(a, b, 2. / 3.);
        const float2 v4 = b;

        outputRecords.Get(0).a = v0;
        outputRecords.Get(0).b = v1;

        outputRecords.Get(1).a = v1;
        outputRecords.Get(1).b = v2;

        outputRecords.Get(2).a = v2;
        outputRecords.Get(2).b = v3;

        outputRecords.Get(3).a = v3;
        outputRecords.Get(3).b = v4;
    }
    else
    {
        // We've reached the recursion limit, thus we draw the current line segment
        // to the output.
        DrawLine(a, b);
    }

    outputRecords.OutputComplete();
}

// [Task 2 Solution]:
[Shader("node")]
    [NodeLaunch("thread")]
    [NodeMaxRecursionDepth(4)]
    [NodeId("Sponge")] void
    SpongeNode(
        ThreadNodeInputRecord<Box> inputRecord,

        [MaxRecords(8)]
            [NodeId("Sponge")] NodeOutput<Box>
                recursiveOutput)
{
    const float2 topLeft = inputRecord.Get().topLeft;
    const float size = inputRecord.Get().size;

    // Check if we have reached the recursion limit.
    const bool hasOutput = GetRemainingRecursionLevels() != 0;

    // Split each box into eight boxes:
    // +---+---+---+
    // | 0 | 1 | 2 |
    // +---+---+---+
    // | 3 |   | 4 |
    // +---+---+---+
    // | 5 | 6 | 7 |
    // +---+---+---+
    ThreadNodeOutputRecords<Box> outputRecords = recursiveOutput.GetThreadNodeOutputRecords(hasOutput * 8);

    if (hasOutput)
    {
        const float newSize = size / 3.;

        uint outputRecordIndex = 0;

        for (uint row = 0; row < 3; ++row)
        {
            for (uint col = 0; col < 3; ++col)
            {
                // Skip the center (see visualization above).
                if (row == 1 && col == 1)
                    continue;

                outputRecords.Get(outputRecordIndex).size = newSize;
                outputRecords.Get(outputRecordIndex).topLeft = topLeft + float2(col * newSize, row * newSize);

                // Advance to next output record.
                outputRecordIndex++;
            }
        }
    }
    else
    {
        // We've reached the recursion limit, thus we draw the current box to the output.
        FillRect(topLeft, topLeft + size);
    }

    outputRecords.OutputComplete();
}