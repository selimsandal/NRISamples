#include "NRI.hlsl"

struct Constants {
    uint elementCount;
    uint multiplier;
    uint addend;
    uint xorMask;
};

NRI_ROOT_CONSTANTS(Constants, gConstants, 0, 0);
#if !USE_HEAP
NRI_RESOURCE(StructuredBuffer<uint>, gInput, t, 0, 0);
NRI_RESOURCE(RWStructuredBuffer<uint>, gOutput, u, 0, 0);
#endif

[numthreads(8, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
#if USE_HEAP
    StructuredBuffer<uint> gInput = ResourceDescriptorHeap[3];
    RWStructuredBuffer<uint> gOutput = ResourceDescriptorHeap[7];
#endif
    uint index = dispatchThreadId.x;
    if (index >= gConstants.elementCount)
        return;

    gOutput[index] = (gInput[index] * gConstants.multiplier + gConstants.addend) ^ gConstants.xorMask;
}
