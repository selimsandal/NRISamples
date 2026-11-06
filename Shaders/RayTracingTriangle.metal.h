// © 2026 NVIDIA Corporation

// Shared by the native Metal versions of the "RayTracingTriangle" shaders

#include "NRI.metal"

// "Sample::CreateRayTracingPipeline": one descriptor set, ranges in declaration order
struct DescriptorSet0 {
    struct {
        ulong bufferAddress;
        metal::texture2d<float, metal::access::write> texture; // "u0", STORAGE_TEXTURE
        ulong metadata;
    } outputImage;

    struct {
        constant NriAccelerationStructureHeader* header; // "t1", ACCELERATION_STRUCTURE
        ulong resourceId;
        ulong metadata;
    } topLevelAS;
};

// No root constants or root descriptors: a pointer to the resource table of set 0, padded to 16 bytes
struct RootData {
    constant DescriptorSet0* set0;
    ulong padding;
};

struct Payload {
    float3 hitValue;
};
