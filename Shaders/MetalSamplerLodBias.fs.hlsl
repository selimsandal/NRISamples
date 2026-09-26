// © 2026 NVIDIA Corporation

#include "NRI.hlsl"

NRI_RESOURCE(Texture2D<float4>, gTexture, t, 0, 0);
NRI_RESOURCE(SamplerState, gSampler, s, 0, 0);

float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    return gTexture.Sample(gSampler, uv);
}
