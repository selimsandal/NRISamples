// © 2026 NVIDIA Corporation

// Native Metal version of "RayTracingTriangle.rgen.hlsl"

#include "RayTracingTriangle.metal.h"

[[visible]] void raygen(thread NriRayContext& context, thread uchar*) {
    constant RootData& root = *(constant RootData*)context.rootData;
    constant DescriptorSet0& set0 = *root.set0;

    const uint2 dispatchRaysIndex = context.dispatchIndex.xy;
    const uint2 dispatchRaysDimensions = context.dispatchDimensions.xy;

    const float2 pixelCenter = float2(dispatchRaysIndex) + float2(0.5, 0.5);
    const float2 inUV = pixelCenter / float2(dispatchRaysDimensions);

    float2 d = inUV * 2.0 - 1.0;
    float aspectRatio = float(dispatchRaysDimensions.x) / float(dispatchRaysDimensions.y);

    metal::raytracing::ray rayDesc;
    rayDesc.origin = float3(0, 0, -2.0);
    rayDesc.direction = metal::normalize(float3(d.x * aspectRatio, -d.y, 1));
    rayDesc.min_distance = 0.001;
    rayDesc.max_distance = 100.0;

    uint rayFlags = NRI_RAY_FLAG_FORCE_OPAQUE;
    uint instanceInclusionMask = 0xff;
    uint rayContributionToHitGroupIndex = 0;
    uint multiplierForGeometryContributionToHitGroupIndex = 1;
    uint missShaderIndex = 0;

    Payload payload = {};
    NriTraceRay(context, *set0.topLevelAS.header, rayFlags, instanceInclusionMask, rayContributionToHitGroupIndex,
        multiplierForGeometryContributionToHitGroupIndex, missShaderIndex, rayDesc, (thread uchar*)&payload);

    set0.outputImage.texture.write(float4(payload.hitValue, 0), dispatchRaysIndex);
}
