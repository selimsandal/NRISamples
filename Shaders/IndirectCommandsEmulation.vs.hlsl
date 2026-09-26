// © 2026 NVIDIA Corporation

#define NRI_ENABLE_DRAW_PARAMETERS_EMULATION
#define NRI_ENABLE_DRAW_INDEX_EMULATION
#include "NRI.hlsl"

NRI_ENABLE_DRAW_PARAMETERS;
RWStructuredBuffer<uint4> Results : register(u0);

struct Output {
    float4 position : SV_Position;
};

Output main(NRI_DECLARE_DRAW_PARAMETERS) {
    const float2 positions[] = {
        float2(-0.5, 0.5),
        float2(0.5, 0.5),
        float2(0.0, -0.5),
    };

    Output output;
    uint vertex = NRI_VERTEX_ID % 3;
    Results[(NRI_BASE_INSTANCE + NRI_DRAW_ID) * 3 + vertex] = uint4(NRI_BASE_VERTEX, NRI_BASE_INSTANCE, NRI_DRAW_ID, NRI_VERTEX_ID_OFFSET);
    output.position = float4(positions[vertex], 0.0, 1.0);

    return output;
}
