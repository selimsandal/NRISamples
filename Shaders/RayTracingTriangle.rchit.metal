// © 2026 NVIDIA Corporation

// Native Metal version of "RayTracingTriangle.rchit.hlsl"

#include "RayTracingTriangle.metal.h"

[[visible]] void closest_hit(thread NriRayContext& context, thread uchar* payloadData) {
    thread Payload& payload = *(thread Payload*)payloadData;

    float2 barycentrics = context.hit.barycentrics;

    payload.hitValue = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
}
