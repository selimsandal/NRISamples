// © 2026 NVIDIA Corporation

// Native Metal version of "RayTracingTriangle.rmiss.hlsl"

#include "RayTracingTriangle.metal.h"

[[visible]] void miss(thread NriRayContext&, thread uchar* payloadData) {
    thread Payload& payload = *(thread Payload*)payloadData;

    payload.hitValue = float3(0.4, 0.3, 0.35);
}
