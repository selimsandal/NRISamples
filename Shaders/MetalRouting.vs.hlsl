// © 2026 NVIDIA Corporation

struct Output {
    float4 position : SV_Position;
    nointerpolation uint instance : TEXCOORD0;
    uint layer : SV_RenderTargetArrayIndex;
    uint viewport : SV_ViewportArrayIndex;
};

Output main(uint vertex : SV_VertexID, uint instance : SV_InstanceID) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    Output result;
    result.position = float4(positions[vertex], 0, 1);
    result.instance = instance;
    result.layer = 1 - instance;
    result.viewport = instance;
    return result;
}
