// © 2026 NVIDIA Corporation

#include "NRI.hlsl"

struct Output {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Output main(uint vertexId : SV_VertexID) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    Output output;
    output.position = float4(positions[vertexId], 0, 1);
    output.uv = positions[vertexId] * float2(0.5, -0.5) + 0.5;
    return output;
}
