#include <metal_stdlib>
#include <metal_raytracing>
#include "NRI.metal"

using namespace metal;

struct Root {
    uint elementCount;
    uint multiplier;
    uint addend;
    uint xorMask;
    const device uint* input;
    device uint* output;
};

kernel void main0(constant Root& root [[buffer(2)]], uint3 dispatchThreadId [[thread_position_in_grid]]) {
    uint index = dispatchThreadId.x;
    if (index >= root.elementCount)
        return;

    root.output[index] = (root.input[index] * root.multiplier + root.addend) ^ root.xorMask;
}

struct ResolveVertex {
    float4 position [[position]];
    uint layer [[flat]];
};

vertex ResolveVertex resolveVertex(uint vertexId [[vertex_id]], uint instanceId [[instance_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[vertexId], 0, 1), instanceId};
}

vertex ResolveVertex depthClampVertex(uint vertexId [[vertex_id]], uint instanceId [[instance_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[vertexId], 1.5f, 1), instanceId};
}

fragment float4 resolveFragment(ResolveVertex input [[stage_in]], uint sampleId [[sample_id]]) {
    return float4(float(sampleId + 1 + input.layer) * 0.125f, float(3 - sampleId) * 0.25f, sampleId == 2 ? 1.0f : 0.0f, 1.0f);
}

fragment float4 samplePositionFragment() {
    return float4(get_sample_position(2), 0.25f, 1.0f);
}

struct MultiviewVertex {
    float4 position [[position]];
    uint amplificationId [[flat]];
    uint targetLayer [[render_target_array_index]];
};

vertex MultiviewVertex multiviewVertex(uint vertexId [[vertex_id]], uint amplificationId [[amplification_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[vertexId], 0, 1), amplificationId, 0};
}

vertex MultiviewVertex multiviewFlexibleVertex(uint vertexId [[vertex_id]], uint amplificationId [[amplification_id]], constant NriMultiview& views [[buffer(3)]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[vertexId], 0, 1), amplificationId, views.viewIndices[amplificationId]};
}

fragment float4 multiviewFragment(MultiviewVertex input [[stage_in]]) {
    return input.amplificationId == 0 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}

struct RoutingVertex {
    float4 position [[position]];
    uint instance [[flat]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
};

vertex RoutingVertex routingVertex(uint vertexId [[vertex_id]], uint instance [[instance_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[vertexId], 0, 1), instance, 1 - instance, instance};
}

fragment float4 routingFragment(RoutingVertex input [[stage_in]]) {
    return input.instance == 0 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}

struct SamplerTextureEntry {
    constant uchar* buffer [[id(0)]];
    texture2d<float> texture [[id(1)]];
    ulong metadata [[id(2)]];
};

struct SamplerEntry {
    sampler value [[id(0)]];
    ulong unused [[id(1)]];
    ulong metadata [[id(2)]];
};

struct SamplerRoot {
    constant SamplerTextureEntry* textures;
    constant SamplerEntry* samplers;
};

struct SamplerVertex {
    float4 position [[position]];
    float2 uv;
};

vertex SamplerVertex samplerLodBiasVertex(uint vertexId [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    const float2 position = positions[vertexId];
    return {float4(position, 0, 1), position * float2(0.5, -0.5) + 0.5};
}

fragment float4 samplerLodBiasFragment(SamplerVertex input [[stage_in]], constant SamplerRoot& root [[buffer(2)]]) {
    return root.textures[0].texture.sample(root.samplers[0].value, input.uv, bias(NriGetSamplerMipBias(root.samplers[0].metadata)));
}

struct VertexStrideInput {
    float2 position [[attribute(0)]];
    float2 instanceOffset [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct VertexStrideOutput {
    float4 position [[position]];
    float4 color;
};

vertex VertexStrideOutput vertexStrideVertex(VertexStrideInput input [[stage_in]]) {
    return {float4(input.position + input.instanceOffset, 0, 1), input.color};
}

fragment float4 vertexStrideFragment(VertexStrideOutput input [[stage_in]]) {
    return input.color;
}

struct DepthStencilViews {
    ulong depthAddress;
    depth2d<float> depth;
    ulong depthMetadata;
    ulong stencilAddress;
    texture2d<uint> stencil;
    ulong stencilMetadata;
};

struct DepthStencilRoot {
    device uint2* output;
    constant DepthStencilViews* views;
};

kernel void readDepthStencil(constant DepthStencilRoot& root [[buffer(2)]], uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= 7 || tid.y >= 5)
        return;

    root.output[tid.y * 7 + tid.x] = uint2(as_type<uint>(root.views->depth.read(tid.xy)), root.views->stencil.read(tid.xy).x);
}

// Native ray dispatch fixture: both records use one signature, with different
// results so an ignored SBT record or incorrect indirect grid is observable.
struct NativeScene {
    raytracing::instance_acceleration_structure accelerationStructure;
    constant uint* instanceContributions;
    ulong reserved[6];
};

static_assert(sizeof(NativeScene) == 64);
static_assert(__builtin_offsetof(NativeScene, instanceContributions) == 8);

using NativeRaygen = void(constant Root&, uint3, uint3);
struct NativeRayArguments {
    NriRayDispatchDesc dispatch;
    constant Root* root;
    constant NriDescriptorEntry* resources;
    constant NriDescriptorEntry* samplers;
    visible_function_table<NativeRaygen> visibleFunctions;
    raytracing::intersection_function_table<> intersectionFunctions;
    ulong intersectionTables;
};

static_assert(sizeof(NativeRayArguments) == sizeof(NriRayDispatchArguments));
static_assert(__builtin_offsetof(NativeRayArguments, visibleFunctions) == __builtin_offsetof(NriRayDispatchArguments, visibleFunctions));
static_assert(__builtin_offsetof(NativeRayArguments, intersectionFunctions) == __builtin_offsetof(NriRayDispatchArguments, intersectionFunctions));

[[visible]] void nativeRaygen(constant Root& root, uint3 position, uint3 size) {
    uint index = (position.z * size.y + position.y) * size.x + position.x;
    root.output[index] = index * root.multiplier + root.addend;
}

[[visible]] void nativeRaygenAlternate(constant Root& root, uint3 position, uint3 size) {
    uint index = (position.z * size.y + position.y) * size.x + position.x;
    root.output[index] = (index * root.multiplier + root.addend) ^ root.xorMask;
}

kernel void RaygenIndirection(constant NativeRayArguments& args [[buffer(3)]], uint3 position [[thread_position_in_grid]]) {
    uint3 size(args.dispatch.width, args.dispatch.height, args.dispatch.depth);
    if (any(position >= size))
        return;

    args.visibleFunctions[args.dispatch.raygen.address->shaderHandle](*args.root, position, size);
}
