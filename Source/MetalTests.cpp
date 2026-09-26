// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include "Extensions/NRIDescriptorHeap.h"
#include "Extensions/NRIRayTracing.h"
#include "Extensions/NRIUpscaler.h"
#include "Extensions/NRIWrapperMetal.h"

#include <algorithm>
#include <array>
#include <vector>

#if METAL_NATIVE_TESTS
#    define NS_PRIVATE_IMPLEMENTATION
#    define MTL_PRIVATE_IMPLEMENTATION
#    include <Metal/Metal.hpp>
#endif

namespace {

constexpr uint32_t ELEMENT_NUM = 73;
constexpr uint32_t THREAD_GROUP_SIZE = 8;
constexpr uint32_t GUARD_NUM = 9;
constexpr uint32_t SENTINEL = 0xD15EA5E5;

struct Constants {
    uint32_t elementCount;
    uint32_t multiplier;
    uint32_t addend;
    uint32_t xorMask;
};

nri::ShaderDesc LoadComputeShader(test::Context& context, const char* fileName, const char* entryPointName = nullptr) {
    context.shaderStorage.emplace_back();
    std::vector<uint8_t>& bytecode = context.shaderStorage.back();
    const std::string path = utils::GetFullPath(fileName, utils::DataFolder::SHADERS);
    if (!utils::LoadFile(path, bytecode))
        return {};

    nri::ShaderDesc shader = {};
    shader.stage = nri::StageBits::COMPUTE_SHADER;
    shader.bytecode = bytecode.data();
    shader.size = bytecode.size();
    shader.entryPointName = entryPointName;

    return shader;
}

bool RunPipeline(test::Context& context, nri::Queue& queue, nri::PipelineLayout& pipelineLayout, const nri::ShaderDesc& shader, const char* name, const std::vector<uint32_t>& inputData, const std::vector<uint32_t>& initialOutput, uint64_t descriptorOffset, uint32_t rootOffset, const Constants& constants, std::vector<uint32_t>& result, bool useHeap = false, bool useMutable = false) {
    nri::ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = &pipelineLayout;
    pipelineDesc.shader = shader;
    nri::Pipeline* pipeline = nullptr;

    struct Cache {
        const nri::CoreInterface& core;
        nri::PipelineCache* object = nullptr;

        ~Cache() {
            core.DestroyPipelineCache(object);
        }
    } cache{context.core};

    // Apple shader instrumentation is incompatible with Metal binary archives.
    const char* shaderValidation = std::getenv("MTL_SHADER_VALIDATION");
    const bool testCache = !shaderValidation || std::strcmp(shaderValidation, "1") != 0;
    if (testCache) {
        const nri::PipelineCacheDesc emptyCache = {};
        TEST_CHECK(context.core.CreatePipelineCache(*context.device, emptyCache, cache.object));
        pipelineDesc.cache = cache.object;
        if (context.deviceDesc->features.pipelineCacheControl) {
            pipelineDesc.flags = nri::ComputePipelineBits::FAIL_ON_CACHE_MISS;
            TEST_CHECK(context.core.CreateComputePipeline(*context.device, pipelineDesc, pipeline) == nri::Result::FAILURE && pipeline == nullptr);
            pipelineDesc.flags = nri::ComputePipelineBits::NONE;
        }
    } else {
        printf("SKIP  pipeline archive round-trip with Metal shader validation\n");
    }
    TEST_CHECK(context.core.CreateComputePipeline(*context.device, pipelineDesc, pipeline));
    context.Track(pipeline);

    if (testCache) {
        uint64_t cacheSize = 0;
        TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, nullptr, cacheSize));
        TEST_CHECK(cacheSize > 0);
        std::vector<uint8_t> cacheData(cacheSize);
        TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, cacheData.data(), cacheSize));
        context.core.DestroyPipelineCache(cache.object);
        cache.object = nullptr;
        const nri::PipelineCacheDesc savedCache = {cacheData.data(), cacheSize};
        TEST_CHECK(context.core.CreatePipelineCache(*context.device, savedCache, cache.object));
        pipelineDesc.cache = cache.object;
        if (context.deviceDesc->features.pipelineCacheControl)
            pipelineDesc.flags = nri::ComputePipelineBits::FAIL_ON_CACHE_MISS;
        TEST_CHECK(context.core.CreateComputePipeline(*context.device, pipelineDesc, pipeline));
        context.Track(pipeline);

        // Saving a loaded archive must preserve its backing data, including repeated size/data queries.
        for (uint32_t i = 0; i < 2; i++) {
            TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, nullptr, cacheSize));
            TEST_CHECK(cacheSize > 0);
            cacheData.resize(cacheSize);
            TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, cacheData.data(), cacheSize));
        }
    }

    const uint64_t bufferSize = descriptorOffset + rootOffset + initialOutput.size() * sizeof(uint32_t);
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = bufferSize;
    bufferDesc.structureStride = sizeof(uint32_t);
    bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;

    nri::Buffer* input = nullptr;
    nri::Buffer* output = nullptr;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, input));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, output));
    bufferDesc.usage = nri::BufferUsageBits::NONE;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));

    std::vector<uint8_t> inputUpload(bufferSize, 0xA7);
    std::vector<uint8_t> outputUpload(bufferSize, 0x5B);
    memcpy(inputUpload.data() + descriptorOffset + rootOffset, inputData.data(), inputData.size() * sizeof(uint32_t));
    memcpy(outputUpload.data() + descriptorOffset + rootOffset, initialOutput.data(), initialOutput.size() * sizeof(uint32_t));

    const nri::BufferUploadDesc uploads[] = {
        {inputUpload.data(), input, {nri::AccessBits::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER}},
        {outputUpload.data(), output, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER}},
    };
    TEST_CHECK(context.helper.UploadData(queue, nullptr, 0, uploads, 2));

    nri::BufferViewDesc viewDesc = {};
    viewDesc.buffer = input;
    viewDesc.type = nri::BufferView::STRUCTURED_BUFFER;
    viewDesc.offset = descriptorOffset;
    viewDesc.size = inputData.size() * sizeof(uint32_t);
    nri::Descriptor* inputView = nullptr;
    TEST_CHECK(context.core.CreateBufferView(viewDesc, inputView));
    context.Track(inputView);

    viewDesc.buffer = output;
    viewDesc.type = nri::BufferView::STORAGE_STRUCTURED_BUFFER;
    nri::Descriptor* outputView = nullptr;
    TEST_CHECK(context.core.CreateBufferView(viewDesc, outputView));
    context.Track(outputView);

    nri::DescriptorPool* mutablePools[2] = {};
    nri::DescriptorSet* mutableSets[2] = {};
    if (useMutable) {
        nri::DescriptorPoolDesc poolDesc = {};
        poolDesc.descriptorSetMaxNum = 1;
        poolDesc.mutableMaxNum = 8;
        poolDesc.flags = nri::DescriptorPoolBits::COPY_SOURCE;
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, mutablePools[0]));
        context.Track(mutablePools[0]);
        poolDesc.flags = nri::DescriptorPoolBits::NONE;
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, mutablePools[1]));
        context.Track(mutablePools[1]);
        TEST_CHECK(context.core.AllocateDescriptorSets(*mutablePools[0], pipelineLayout, 0, &mutableSets[0], 1, 0));
        TEST_CHECK(context.core.AllocateDescriptorSets(*mutablePools[1], pipelineLayout, 0, &mutableSets[1], 1, 0));

        // Exercise both type changes and copying at the sparse indices consumed by ResourceDescriptorHeap.
        const nri::UpdateDescriptorRangeDesc updates[] = {
            {mutableSets[0], 0, 3, &outputView, 1},
            {mutableSets[0], 0, 7, &inputView, 1},
            {mutableSets[0], 0, 3, &inputView, 1},
            {mutableSets[0], 0, 7, &outputView, 1},
        };
        context.core.UpdateDescriptorRanges(updates, 4);
        const nri::CopyDescriptorRangeDesc copies[] = {
            {mutableSets[1], 0, 3, mutableSets[0], 0, 3, 1},
            {mutableSets[1], 0, 7, mutableSets[0], 0, 7, 1},
        };
        context.core.CopyDescriptorRanges(copies, 2);
    }

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, useMutable ? mutablePools[1] : nullptr));
    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, pipelineLayout);
    context.core.CmdSetPipeline(*commandBuffer, *pipeline);

    const nri::SetRootConstantsDesc rootConstants = {0, &constants, sizeof(constants), 0, nri::BindPoint::COMPUTE};
    context.core.CmdSetRootConstants(*commandBuffer, rootConstants);

    struct Heap {
        nri::DescriptorHeapInterface interface = {};
        nri::DescriptorHeap* object = nullptr;

        ~Heap() {
            if (object)
                interface.DestroyDescriptorHeap(object);
        }
    };

    Heap heap;
    if (useHeap) {
        TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::DescriptorHeapInterface), &heap.interface));
        TEST_CHECK(heap.interface.CreateDescriptorHeap(*context.device, {9, 0}, heap.object));
        const nri::WriteResourceDescriptorsDesc writes[] = {{outputView, 7}, {inputView, 3}};
        TEST_CHECK(heap.interface.WriteResourceDescriptors(*heap.object, writes, 2));
        heap.interface.CmdSetDescriptorHeap(*commandBuffer, *heap.object);
    } else if (useMutable) {
        context.core.CmdSetDescriptorSet(*commandBuffer, {0, mutableSets[1], nri::BindPoint::COMPUTE});
    } else {
        const nri::SetRootDescriptorDesc rootDescriptors[] = {
            {0, inputView, rootOffset, nri::BindPoint::COMPUTE},
            {1, outputView, rootOffset, nri::BindPoint::COMPUTE},
        };
        context.core.CmdSetRootDescriptor(*commandBuffer, rootDescriptors[0]);
        context.core.CmdSetRootDescriptor(*commandBuffer, rootDescriptors[1]);
    }
    context.core.CmdDispatch(*commandBuffer, {uint32_t((ELEMENT_NUM + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE), 1, 1});

    nri::BufferBarrierDesc barrier = {};
    barrier.buffer = output;
    barrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
    barrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.buffers = &barrier;
    barrierDesc.bufferNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    context.core.CmdCopyBuffer(*commandBuffer, *readback, 0, *output, 0, nri::WHOLE_SIZE);
    TEST_CHECK(context.core.EndCommandBuffer(*commandBuffer));

    nri::Fence* fence = nullptr;
    TEST_CHECK(context.core.CreateFence(*context.device, 0, fence));
    context.Track(fence);
    nri::FenceSubmitDesc signalFence = {fence, 1, nri::StageBits::ALL};
    nri::QueueSubmitDesc submitDesc = {};
    submitDesc.commandBuffers = &commandBuffer;
    submitDesc.commandBufferNum = 1;
    submitDesc.signalFences = &signalFence;
    submitDesc.signalFenceNum = 1;
    TEST_CHECK(context.core.QueueSubmit(queue, submitDesc));
    context.core.Wait(*fence, 1);
    TEST_CHECK(context.core.GetFenceValue(*fence) >= 1);

    const uint8_t* mapped = (const uint8_t*)context.core.MapBuffer(*readback, 0, bufferSize);
    TEST_CHECK(mapped != nullptr);
    bool passed = true;
    for (uint64_t i = 0; i < descriptorOffset + rootOffset; i++)
        passed &= mapped[i] == 0x5B;

    const uint32_t* outputData = (const uint32_t*)(mapped + descriptorOffset + rootOffset);
    result.assign(outputData, outputData + initialOutput.size());
    context.core.UnmapBuffer(*readback);

    for (uint32_t i = 0; i < ELEMENT_NUM; i++) {
        const uint32_t expected = (inputData[i] * constants.multiplier + constants.addend) ^ constants.xorMask;
        passed &= result[i] == expected;
    }
    for (uint32_t i = ELEMENT_NUM; i < result.size(); i++)
        passed &= result[i] == SENTINEL;

    return test::Report(name, passed);
}

bool TestBufferClears(test::Context& context, nri::Queue& queue) {
    constexpr uint32_t typedPrefixNum = 4;
    constexpr uint32_t typedElementNum = 5;
    constexpr uint32_t typedSuffixNum = 3;
    constexpr uint32_t structuredPrefixNum = 3;
    constexpr uint32_t structuredElementNum = 4;
    constexpr uint32_t structuredSuffixNum = 2;
    constexpr uint32_t structureStride = 16;

    const uint64_t typedOffset = std::max<uint64_t>(context.deviceDesc->memoryAlignment.bufferShaderResourceOffset, typedPrefixNum * 16);
    const uint64_t typedSize = typedOffset + (typedElementNum + typedSuffixNum) * 16;
    const uint64_t structuredOffset = std::max<uint64_t>(context.deviceDesc->memoryAlignment.bufferShaderResourceOffset, structuredPrefixNum * structureStride);
    const uint64_t structuredSize = structuredOffset + (structuredElementNum + structuredSuffixNum) * structureStride;
    std::vector<uint32_t> typedInitial(typedSize / sizeof(uint32_t));
    std::vector<uint32_t> structuredInitial(structuredSize / sizeof(uint32_t));
    for (uint32_t i = 0; i < typedInitial.size(); i++)
        typedInitial[i] = 0x81000000u + i * 0x10101u;
    for (uint32_t i = 0; i < structuredInitial.size(); i++)
        structuredInitial[i] = 0x42000000u + i * 0x10003u;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = typedSize;
    bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;
    nri::Buffer* typed = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, typed));
    bufferDesc.size = structuredSize;
    bufferDesc.structureStride = structureStride;
    nri::Buffer* structured = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, structured));
    bufferDesc.usage = nri::BufferUsageBits::NONE;
    bufferDesc.size = typedSize;
    nri::Buffer* typedReadback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, typedReadback));
    bufferDesc.size = structuredSize;
    nri::Buffer* structuredReadback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, structuredReadback));

    const nri::BufferUploadDesc uploads[] = {
        {typedInitial.data(), typed, {nri::AccessBits::CLEAR_STORAGE, nri::StageBits::CLEAR_STORAGE}},
        {structuredInitial.data(), structured, {nri::AccessBits::CLEAR_STORAGE, nri::StageBits::CLEAR_STORAGE}},
    };
    TEST_CHECK(context.helper.UploadData(queue, nullptr, 0, uploads, 2));

    nri::BufferViewDesc viewDesc = {};
    viewDesc.buffer = typed;
    viewDesc.type = nri::BufferView::STORAGE_BUFFER;
    viewDesc.offset = typedOffset;
    viewDesc.size = typedElementNum * 16;
    viewDesc.format = nri::Format::RGBA32_UINT;
    nri::Descriptor* typedView = nullptr;
    TEST_CHECK(context.core.CreateBufferView(viewDesc, typedView));
    context.Track(typedView);

    viewDesc.buffer = structured;
    viewDesc.type = nri::BufferView::STORAGE_STRUCTURED_BUFFER;
    viewDesc.offset = structuredOffset;
    viewDesc.size = structuredElementNum * structureStride;
    viewDesc.format = nri::Format::UNKNOWN;
    viewDesc.structureStride = structureStride;
    nri::Descriptor* structuredView = nullptr;
    TEST_CHECK(context.core.CreateBufferView(viewDesc, structuredView));
    context.Track(structuredView);

    const nri::DescriptorRangeDesc ranges[] = {
        {0, 1, nri::DescriptorType::STORAGE_BUFFER, nri::StageBits::COMPUTE_SHADER},
        {1, 1, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
    };
    nri::DescriptorSetDesc setDesc = {};
    setDesc.ranges = ranges;
    setDesc.rangeNum = 2;
    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = &setDesc;
    pipelineLayoutDesc.descriptorSetNum = 1;
    pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);
    nri::DescriptorPoolDesc poolDesc = {};
    poolDesc.descriptorSetMaxNum = 1;
    poolDesc.storageBufferMaxNum = 1;
    poolDesc.storageStructuredBufferMaxNum = 1;
    nri::DescriptorPool* pool = nullptr;
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, pool));
    context.Track(pool);
    nri::DescriptorSet* set = nullptr;
    TEST_CHECK(context.core.AllocateDescriptorSets(*pool, *pipelineLayout, 0, &set, 1, 0));
    const nri::UpdateDescriptorRangeDesc updates[] = {
        {set, 0, 0, &typedView, 1},
        {set, 1, 0, &structuredView, 1},
    };
    context.core.UpdateDescriptorRanges(updates, 2);

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, pool));
    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *pipelineLayout);
    context.core.CmdSetDescriptorSet(*commandBuffer, {0, set, nri::BindPoint::COMPUTE});
    nri::ClearStorageDesc clear = {};
    clear.descriptor = typedView;
    clear.value.ui = {0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001};
    context.core.CmdClearStorage(*commandBuffer, clear);
    clear.descriptor = structuredView;
    clear.value.ui = {0x13579BDF, 0x2468ACE0, 0xDEADBEEF, 0x01020304};
    clear.rangeIndex = 1;
    context.core.CmdClearStorage(*commandBuffer, clear);

    nri::BufferBarrierDesc barriers[2] = {};
    barriers[0].buffer = typed;
    barriers[0].before = {nri::AccessBits::CLEAR_STORAGE, nri::StageBits::CLEAR_STORAGE};
    barriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    barriers[1] = barriers[0];
    barriers[1].buffer = structured;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.buffers = barriers;
    barrierDesc.bufferNum = 2;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    context.core.CmdCopyBuffer(*commandBuffer, *typedReadback, 0, *typed, 0, nri::WHOLE_SIZE);
    context.core.CmdCopyBuffer(*commandBuffer, *structuredReadback, 0, *structured, 0, nri::WHOLE_SIZE);
    TEST_CHECK(context.SubmitAndWait(queue, *commandBuffer));

    const uint32_t* typedData = (const uint32_t*)context.core.MapBuffer(*typedReadback, 0, typedSize);
    const uint32_t* structuredData = (const uint32_t*)context.core.MapBuffer(*structuredReadback, 0, structuredSize);
    TEST_CHECK(typedData && structuredData);
    bool typedPassed = true;
    const std::array<uint32_t, 4> typedExpected = {0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001};
    for (uint32_t i = 0; i < typedInitial.size(); i++) {
        const bool inView = i >= typedOffset / 4 && i < typedOffset / 4 + typedElementNum * 4;
        typedPassed &= typedData[i] == (inView ? typedExpected[(i - typedOffset / 4) % 4] : typedInitial[i]);
    }
    bool structuredPassed = true;
    for (uint32_t i = 0; i < structuredInitial.size(); i++) {
        const bool inView = i >= structuredOffset / 4 && i < structuredOffset / 4 + structuredElementNum * structureStride / 4;
        // NRI specifies that buffer clears repeat the first clear component across the view.
        structuredPassed &= structuredData[i] == (inView ? 0x13579BDFu : structuredInitial[i]);
    }
    context.core.UnmapBuffer(*typedReadback);
    context.core.UnmapBuffer(*structuredReadback);

    const bool typedReported = test::Report("CmdClearStorage typed RGBA32_UINT buffer", typedPassed);
    const bool structuredReported = test::Report("CmdClearStorage structured buffer", structuredPassed);

    return typedReported && structuredReported;
}

bool TestTextureClear(test::Context& context, nri::Queue& queue, nri::Format format, const nri::Color& clearValue, const char* name, nri::TextureType type = nri::TextureType::TEXTURE_2D) {
    const bool volume = type == nri::TextureType::TEXTURE_3D;
    constexpr uint16_t width = 8;
    const uint16_t height = type == nri::TextureType::TEXTURE_1D ? 1 : 6;
    const uint16_t depth = volume ? 8 : 1;
    constexpr uint16_t mipNum = 2;
    const uint16_t layerNum = volume ? 1 : 2;
    constexpr uint16_t targetMip = 1;
    constexpr uint16_t targetLayer = 1;
    constexpr uint32_t texelSize = 4;
    constexpr uint32_t targetWidth = width >> targetMip;
    const uint32_t targetHeight = std::max(1, height >> targetMip);
    const uint32_t readbackSlices = volume ? depth >> targetMip : layerNum;

    std::vector<std::vector<uint32_t>> initial(mipNum * layerNum);
    std::vector<nri::TextureSubresourceUploadDesc> subresources(mipNum * layerNum);
    for (uint32_t layer = 0; layer < layerNum; layer++) {
        for (uint32_t mip = 0; mip < mipNum; mip++) {
            const uint32_t index = layer * mipNum + mip;
            const uint32_t mipWidth = width >> mip;
            const uint32_t mipHeight = std::max(1, height >> mip);
            const uint32_t mipDepth = std::max(1, depth >> mip);
            initial[index].resize(mipWidth * mipHeight * mipDepth);
            for (uint32_t i = 0; i < initial[index].size(); i++)
                initial[index][i] = 0x3E800000u + index * 0x00110000u + i * 0x101u;
            subresources[index] = {initial[index].data(), mipDepth, mipWidth * texelSize, mipWidth * mipHeight * texelSize};
        }
    }

    nri::TextureDesc textureDesc = {};
    textureDesc.type = type;
    textureDesc.format = format;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.depth = depth;
    textureDesc.mipNum = mipNum;
    textureDesc.layerNum = layerNum;
    textureDesc.sampleNum = 1;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));
    const nri::TextureUploadDesc upload = {subresources.data(), texture, {nri::AccessBits::CLEAR_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::CLEAR_STORAGE}, nri::PlaneBits::COLOR};
    TEST_CHECK(context.helper.UploadData(queue, &upload, 1, nullptr, 0));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = texture;
    viewDesc.type = nri::TextureView::STORAGE_TEXTURE;
    viewDesc.format = format;
    viewDesc.mipOffset = targetMip;
    viewDesc.mipNum = 1;
    viewDesc.layerOffset = volume ? 0 : targetLayer;
    viewDesc.layerNum = 1;
    viewDesc.sliceOffset = volume ? 1 : 0;
    viewDesc.sliceNum = volume ? nri::REMAINING : 1;
    viewDesc.planes = nri::PlaneBits::COLOR;
    nri::Descriptor* view = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, view));
    context.Track(view);

    const nri::DescriptorRangeDesc range = {0, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER};
    nri::DescriptorSetDesc setDesc = {};
    setDesc.ranges = &range;
    setDesc.rangeNum = 1;
    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = &setDesc;
    pipelineLayoutDesc.descriptorSetNum = 1;
    pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);
    nri::DescriptorPoolDesc poolDesc = {};
    poolDesc.descriptorSetMaxNum = 1;
    poolDesc.storageTextureMaxNum = 1;
    nri::DescriptorPool* pool = nullptr;
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, pool));
    context.Track(pool);
    nri::DescriptorSet* set = nullptr;
    TEST_CHECK(context.core.AllocateDescriptorSets(*pool, *pipelineLayout, 0, &set, 1, 0));
    const nri::UpdateDescriptorRangeDesc update = {set, 0, 0, &view, 1};
    context.core.UpdateDescriptorRanges(&update, 1);

    const uint32_t rowAlignment = std::max(context.deviceDesc->memoryAlignment.uploadBufferTextureRow, 1u);
    const uint32_t sliceAlignment = std::max(context.deviceDesc->memoryAlignment.uploadBufferTextureSlice, 1u);
    nri::TextureDataLayoutDesc layout = {};
    layout.rowPitch = (targetWidth * texelSize + rowAlignment - 1) / rowAlignment * rowAlignment;
    layout.slicePitch = layout.rowPitch * targetHeight;
    while (layout.slicePitch % sliceAlignment)
        layout.slicePitch += layout.rowPitch;
    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = layout.slicePitch * readbackSlices;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, pool));
    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *pipelineLayout);
    context.core.CmdSetDescriptorSet(*commandBuffer, {0, set, nri::BindPoint::COMPUTE});
    nri::ClearStorageDesc clear = {};
    clear.descriptor = view;
    clear.value = clearValue;
    context.core.CmdClearStorage(*commandBuffer, clear);
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = texture;
    barrier.before = {nri::AccessBits::CLEAR_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::CLEAR_STORAGE};
    barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    barrier.mipOffset = targetMip;
    barrier.mipNum = 1;
    barrier.layerOffset = 0;
    barrier.layerNum = layerNum;
    barrier.planes = nri::PlaneBits::COLOR;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &barrier;
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    nri::TextureRegionDesc region = {};
    region.width = nri::WHOLE_SIZE;
    region.height = nri::WHOLE_SIZE;
    region.depth = 1;
    region.mipOffset = targetMip;
    region.planes = nri::PlaneBits::COLOR;
    for (uint32_t slice = 0; slice < readbackSlices; slice++) {
        region.layerOffset = volume ? 0 : slice;
        region.z = volume ? slice : 0;
        layout.offset = slice * layout.slicePitch;
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, layout, *texture, region);
    }
    TEST_CHECK(context.SubmitAndWait(queue, *commandBuffer));

    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, readbackDesc.size);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t slice = 0; slice < readbackSlices; slice++) {
        for (uint32_t y = 0; y < targetHeight; y++) {
            const uint32_t* row = (const uint32_t*)(data + slice * layout.slicePitch + y * layout.rowPitch);
            for (uint32_t x = 0; x < targetWidth; x++) {
                const bool cleared = volume ? slice >= 1 : slice == targetLayer;
                const uint32_t original = initial[volume ? targetMip : slice * mipNum + targetMip][((volume ? slice : 0) * targetHeight + y) * targetWidth + x];
                passed &= row[x] == (cleared ? clearValue.ui.x : original);
            }
        }
    }
    context.core.UnmapBuffer(*readback);

    return test::Report(name, passed);
}

bool TestResolve(test::Context& context, nri::Queue& queue, uint32_t mode) {
    if ((mode == 11 || mode == 12) && !context.deviceDesc->features.shaderBytecodeDXIL)
        return true;

    const char* cases[] = {"direct", "multi-draw", "count zero", "count one", "count clamped", "indexed multi-draw", "indexed count zero", "indexed count one", "indexed count clamped", "depth clipping", "depth clamping", "DXIL partial sample mask", "DXIL default sample mask (zero means ALL)", "programmable sample positions"};
    printf("Resolve/draw case: %s\n", cases[mode]);
    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, pipelineLayout));
    context.Track(pipelineLayout);
    nri::ShaderDesc shaders[] = {
        LoadComputeShader(context, "MetalTests.metallib", mode == 9 || mode == 10 ? "depthClampVertex" : "resolveVertex"),
        mode == 11 || mode == 12 ? LoadComputeShader(context, "GraphicsPipelineStates.fs.dxil") : LoadComputeShader(context, "MetalTests.metallib", mode == 13 ? "samplePositionFragment" : "resolveFragment"),
    };
    shaders[0].stage = nri::StageBits::VERTEX_SHADER;
    shaders[1].stage = nri::StageBits::FRAGMENT_SHADER;
    nri::ColorAttachmentDesc color = {};
    color.format = nri::Format::RGBA8_UNORM;
    color.colorWriteMask = nri::ColorWriteBits::RGBA;
    const nri::MultisampleDesc multisample = {mode == 11 ? 0x5u : (mode == 12 ? 0u : nri::ALL), 4, false, mode == 13};
    nri::GraphicsPipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = pipelineLayout;
    pipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    pipelineDesc.multisample = &multisample;
    pipelineDesc.rasterization.depthClamp = mode == 10;
    pipelineDesc.outputMerger.colors = &color;
    pipelineDesc.outputMerger.colorNum = 1;
    pipelineDesc.shaders = shaders;
    pipelineDesc.shaderNum = 2;
    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, pipeline));
    context.Track(pipeline);

    const bool indexed = mode >= 5 && mode <= 8;
    std::array<uint8_t, 16 + 2 * 2 * 32> arguments = {};
    for (uint32_t layer = 0; layer < 2; layer++) {
        for (uint32_t draw = 0; draw < 2; draw++) {
            uint8_t* dst = arguments.data() + 16 + layer * 64 + draw * 32;
            if (indexed) {
                const nri::DrawIndexedDesc argument = {3, 1, 1, -1, layer + 2 * draw};
                memcpy(dst, &argument, sizeof(argument));
            } else {
                const nri::DrawDesc argument = {3, 1, 0, layer + 2 * draw};
                memcpy(dst, &argument, sizeof(argument));
            }
        }
    }
    const uint32_t counts[] = {99, 0, 1, 5};
    const uint16_t indices[] = {999, 999, 999, 1, 2, 3};
    nri::BufferDesc argumentDesc = {};
    argumentDesc.size = arguments.size();
    argumentDesc.usage = nri::BufferUsageBits::ARGUMENT;
    nri::Buffer* argumentBuffer = nullptr;
    nri::Buffer* countBuffer = nullptr;
    nri::Buffer* indexBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentDesc, nri::MemoryLocation::DEVICE, argumentBuffer));
    argumentDesc.size = sizeof(counts);
    TEST_CHECK(context.CreateBuffer(argumentDesc, nri::MemoryLocation::DEVICE, countBuffer));
    argumentDesc.size = sizeof(indices);
    argumentDesc.usage = nri::BufferUsageBits::INDEX;
    TEST_CHECK(context.CreateBuffer(argumentDesc, nri::MemoryLocation::DEVICE, indexBuffer));
    const nri::BufferUploadDesc uploads[] = {
        {arguments.data(), argumentBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
        {counts, countBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
        {indices, indexBuffer, {nri::AccessBits::INDEX_BUFFER, nri::StageBits::INDEX_INPUT}},
    };
    TEST_CHECK(context.helper.UploadData(queue, nullptr, 0, uploads, 3));

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = color.format;
    textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.width = 7;
    textureDesc.height = 5;
    textureDesc.layerNum = 2;
    textureDesc.sampleNum = 4;
    nri::Texture* source = nullptr;
    nri::Texture* destination = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, source));
    textureDesc.sampleNum = 1;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, destination));

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = 256 * 5 * 2;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));
    nri::TextureBarrierDesc barriers[2] = {};
    barriers[0].texture = source;
    barriers[0].mipNum = 1;
    barriers[0].layerNum = 2;
    barriers[0].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = barriers;
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    for (uint16_t layer = 0; layer < 2; layer++) {
        nri::TextureViewDesc viewDesc = {};
        viewDesc.texture = source;
        viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
        viewDesc.format = color.format;
        viewDesc.mipNum = 1;
        viewDesc.layerOffset = layer;
        viewDesc.layerNum = 1;
        viewDesc.sliceNum = 1;
        nri::Descriptor* view = nullptr;
        TEST_CHECK(context.core.CreateTextureView(viewDesc, view));
        context.Track(view);
        nri::AttachmentDesc attachment = {};
        attachment.descriptor = view;
        attachment.loadOp = nri::LoadOp::CLEAR;
        attachment.storeOp = nri::StoreOp::STORE;
        nri::RenderingDesc rendering = {};
        rendering.colors = &attachment;
        rendering.colorNum = 1;
        context.core.CmdBeginRendering(*commandBuffer, rendering);
        context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *pipelineLayout);
        context.core.CmdSetPipeline(*commandBuffer, *pipeline);
        const nri::Viewport viewport = {0, 0, 7, 5, 0, 1};
        const nri::Rect scissor = {0, 0, 7, 5};
        context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
        context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
        if (mode == 13) {
            const nri::SampleLocation locations[] = {{-6, -2}, {-4, 6}, {int8_t(layer ? -4 : 2), int8_t(layer ? 4 : -6)}, {6, 2}};
            context.core.CmdSetSampleLocations(*commandBuffer, locations, 4, 4);
        }
        const uint32_t indirectMode = mode >= 9 ? 0 : (indexed ? mode - 4 : mode);
        const nri::Buffer* count = indirectMode >= 2 ? countBuffer : nullptr;
        const uint64_t countOffset = indirectMode >= 2 ? (indirectMode - 1) * sizeof(uint32_t) : 0;
        if (!mode || mode >= 9)
            context.core.CmdDraw(*commandBuffer, {3, 1, 0, layer});
        else if (indexed) {
            context.core.CmdSetIndexBuffer(*commandBuffer, *indexBuffer, 4, nri::IndexType::UINT16);
            context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, 16 + layer * 64, 2, 32, count, countOffset);
        } else
            context.core.CmdDrawIndirect(*commandBuffer, *argumentBuffer, 16 + layer * 64, 2, 32, count, countOffset);
        context.core.CmdEndRendering(*commandBuffer);
    }

    barriers[0].before = barriers[0].after;
    barriers[0].after = {nri::AccessBits::RESOLVE_SOURCE, nri::Layout::RESOLVE_SOURCE, nri::StageBits::RESOLVE};
    barriers[1] = barriers[0];
    barriers[1].texture = destination;
    barriers[1].before = {};
    barriers[1].after = {nri::AccessBits::RESOLVE_DESTINATION, nri::Layout::RESOLVE_DESTINATION, nri::StageBits::RESOLVE};
    barrierDesc.textureNum = 2;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    context.core.CmdResolveTexture(*commandBuffer, *destination, nullptr, *source, nullptr, nri::ResolveOp::AVERAGE);
    barriers[0] = barriers[1];
    barriers[0].before = barriers[0].after;
    barriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    for (uint16_t layer = 0; layer < 2; layer++) {
        nri::TextureRegionDesc region = {};
        region.layerOffset = layer;
        region.width = 7;
        region.height = 5;
        region.depth = 1;
        const nri::TextureDataLayoutDesc dataLayout = {uint64_t(layer) * 1280, 256, 1280};
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, dataLayout, *destination, region);
    }
    TEST_CHECK(context.SubmitAndWait(queue, *commandBuffer));
    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t layer = 0; layer < 2; layer++) {
        const bool zero = mode == 2 || mode == 6 || mode == 9;
        const bool two = mode == 1 || mode == 4 || mode == 5 || mode == 8;
        std::array<int, 4> expected = {two ? (layer ? 175 : 143) : (layer ? 112 : 80), 96, 64, 255};
        if (zero)
            expected = {0, 0, 0, 0};
        else if (mode == 11)
            expected = {102, 38, 13, 128}; // Half of float4(0.8, 0.3, 0.1, 1), converted to UNORM8.
        else if (mode == 12)
            expected = {204, 77, 26, 255};
        else if (mode == 13)
            expected = {layer ? 64 : 159, layer ? 191 : 32, 64, 255};

        for (uint32_t y = 0; y < 5; y++) {
            for (uint32_t x = 0; x < 7; x++) {
                for (uint32_t c = 0; c < 4; c++)
                    passed &= abs(int(data[layer * 1280 + y * 256 + x * 4 + c]) - expected[c]) <= 1;
            }
        }
    }
    context.core.UnmapBuffer(*readback);

    return test::Report("explicit 4x MSAA resolve, per-sample colors and two layers", passed);
}

bool TestNIS(test::Context& context, nri::Queue& queue) {
    nri::UpscalerInterface upscalerInterface = {};
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::UpscalerInterface), &upscalerInterface));
    if (!upscalerInterface.IsUpscalerSupported(*context.device, nri::UpscalerType::NIS)) {
        printf("SKIP  NIS upscaler is unsupported\n");

        return true;
    }

    constexpr uint16_t inputWidth = 13;
    constexpr uint16_t inputHeight = 7;
    constexpr uint16_t outputWidth = 19;
    constexpr uint16_t outputHeight = 11;
    constexpr uint32_t rowPitch = 256;
    constexpr std::array<uint8_t, 4> inputColor = {51, 102, 153, 255};

    nri::UpscalerDesc upscalerDesc = {};
    upscalerDesc.upscaleResolution = {outputWidth, outputHeight};
    upscalerDesc.type = nri::UpscalerType::NIS;
    upscalerDesc.outputFormat = nri::Format::RGBA8_UNORM;
    nri::Upscaler* upscaler = nullptr;
    TEST_CHECK(upscalerInterface.CreateUpscaler(*context.device, upscalerDesc, upscaler));

    struct UpscalerDestroyer {
        const nri::UpscalerInterface& interface;
        nri::Upscaler* upscaler;

        ~UpscalerDestroyer() {
            interface.DestroyUpscaler(upscaler);
        }
    } upscalerDestroyer{upscalerInterface, upscaler};

    std::vector<uint8_t> inputData(inputWidth * inputHeight * 4);
    for (size_t i = 0; i < inputData.size(); i += 4)
        memcpy(inputData.data() + i, inputColor.data(), inputColor.size());

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
    textureDesc.width = inputWidth;
    textureDesc.height = inputHeight;
    nri::Texture* input = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, input));
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
    textureDesc.width = outputWidth;
    textureDesc.height = outputHeight;
    nri::Texture* output = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, output));

    const nri::TextureSubresourceUploadDesc inputSubresource = {inputData.data(), 1, inputWidth * 4, inputWidth * inputHeight * 4};
    const nri::TextureUploadDesc inputUpload = {&inputSubresource, input, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER}, nri::PlaneBits::COLOR};
    TEST_CHECK(context.helper.UploadData(queue, &inputUpload, 1, nullptr, 0));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = input;
    viewDesc.type = nri::TextureView::TEXTURE;
    viewDesc.format = textureDesc.format;
    viewDesc.mipNum = 1;
    viewDesc.layerNum = 1;
    viewDesc.sliceNum = 1;
    nri::Descriptor* inputView = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, inputView));
    context.Track(inputView);
    viewDesc.texture = output;
    viewDesc.type = nri::TextureView::STORAGE_TEXTURE;
    nri::Descriptor* outputView = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, outputView));
    context.Track(outputView);

    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = rowPitch * outputHeight;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commands = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
    TEST_CHECK(context.core.BeginCommandBuffer(*commands, nullptr));
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = output;
    barrier.mipNum = 1;
    barrier.layerNum = 1;
    barrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
    nri::BarrierDesc barriers = {};
    barriers.textures = &barrier;
    barriers.textureNum = 1;
    context.core.CmdBarrier(*commands, barriers);

    nri::DispatchUpscaleDesc dispatchDesc = {};
    dispatchDesc.output = {output, outputView};
    dispatchDesc.input = {input, inputView};
    dispatchDesc.settings.nis.sharpness = 0.0f;
    dispatchDesc.currentResolution = {inputWidth, inputHeight};
    upscalerInterface.CmdDispatchUpscale(*commands, *upscaler, dispatchDesc);

    barrier.before = barrier.after;
    barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    context.core.CmdBarrier(*commands, barriers);
    const nri::TextureDataLayoutDesc layout = {0, rowPitch, rowPitch * outputHeight};
    nri::TextureRegionDesc region = {};
    region.width = outputWidth;
    region.height = outputHeight;
    region.depth = 1;
    context.core.CmdReadbackTextureToBuffer(*commands, *readback, layout, *output, region);
    TEST_CHECK(context.SubmitAndWait(queue, *commands));

    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t y = 0; y < outputHeight; y++) {
        for (uint32_t x = 0; x < outputWidth; x++) {
            for (uint32_t c = 0; c < 3; c++)
                passed &= abs(int(data[y * rowPitch + x * 4 + c]) - int(inputColor[c])) <= 2;
        }
    }
    if (!passed)
        printf("NIS first output RGB: %u,%u,%u (expected %u,%u,%u)\n", data[0], data[1], data[2], inputColor[0], inputColor[1], inputColor[2]);
    context.core.UnmapBuffer(*readback);

    return test::Report("NIS asymmetric upscale RGB readback", passed);
}

bool TestNativeLayerBasedMultiview(test::Context& context, nri::Queue& queue) {
    if (!context.deviceDesc->features.layerBasedMultiview || context.deviceDesc->other.viewMaxNum < 2) {
        printf("SKIP  native Metal layer-based multiview is unsupported\n");

        return true;
    }

    constexpr uint32_t width = 4;
    constexpr uint32_t height = 3;
    constexpr uint32_t rowPitch = 256;
    constexpr uint32_t slicePitch = rowPitch * height;

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    nri::ShaderDesc shaders[] = {
        LoadComputeShader(context, "MetalTests.metallib", "multiviewVertex"),
        LoadComputeShader(context, "MetalTests.metallib", "multiviewFragment"),
    };
    shaders[0].stage = nri::StageBits::VERTEX_SHADER;
    shaders[1].stage = nri::StageBits::FRAGMENT_SHADER;

    nri::ColorAttachmentDesc color = {};
    color.format = nri::Format::RGBA8_UNORM;
    color.colorWriteMask = nri::ColorWriteBits::RGBA;

    const uint32_t viewMasks[] = {3, 2, 3, 2, 0, 0};
    const char* names[] = {
        "native Metal layer-based multiview readback",
        "native Metal sparse layer-based multiview mapping",
        "native Metal viewport-based multiview readback",
        "native Metal flexible multiview subset mapping",
        "native shader viewport/layer routing",
        "converted shader viewport/layer routing",
    };
    bool passed = true;
    for (uint32_t caseIndex = 0; caseIndex < (context.deviceDesc->features.shaderBytecodeDXIL ? 6u : 5u); caseIndex++) {
        const bool routing = caseIndex >= 4;
        const bool viewportBased = caseIndex == 2;
        const bool flexible = caseIndex == 3;
        const uint16_t layerNum = viewportBased ? 1 : 2;
        if (flexible) {
            shaders[0] = LoadComputeShader(context, "MetalTests.metallib", "multiviewFlexibleVertex");
            shaders[0].stage = nri::StageBits::VERTEX_SHADER;
        }
        if (routing) {
            shaders[0] = LoadComputeShader(context, caseIndex == 4 ? "MetalTests.metallib" : "MetalRouting.vs.dxil", caseIndex == 4 ? "routingVertex" : "main");
            shaders[1] = LoadComputeShader(context, caseIndex == 4 ? "MetalTests.metallib" : "MetalRouting.fs.dxil", caseIndex == 4 ? "routingFragment" : "main");
            shaders[0].stage = nri::StageBits::VERTEX_SHADER;
            shaders[1].stage = nri::StageBits::FRAGMENT_SHADER;
        }
        nri::GraphicsPipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = pipelineLayout;
        pipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
        pipelineDesc.outputMerger.colors = &color;
        pipelineDesc.outputMerger.colorNum = 1;
        pipelineDesc.outputMerger.viewMask = flexible ? 3 : (viewportBased ? 0 : viewMasks[caseIndex]);
        pipelineDesc.outputMerger.multiview = flexible ? nri::Multiview::FLEXIBLE : (viewportBased ? nri::Multiview::VIEWPORT_BASED : nri::Multiview::LAYER_BASED);
        pipelineDesc.shaders = shaders;
        pipelineDesc.shaderNum = 2;
        nri::Pipeline* pipeline = nullptr;
        if (flexible) {
            TEST_CHECK(context.deviceDesc->features.flexibleMultiview);
            if (context.deviceDesc->features.shaderBytecodeDXIL) {
                nri::ShaderDesc convertedShaders[] = {LoadComputeShader(context, "TriangleFlexibleMultiview.vs.dxil"), shaders[1]};
                convertedShaders[0].stage = nri::StageBits::VERTEX_SHADER;
                TEST_CHECK(convertedShaders[0].bytecode != nullptr);
                nri::GraphicsPipelineDesc convertedDesc = pipelineDesc;
                convertedDesc.shaders = convertedShaders;
                TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, convertedDesc, pipeline) == nri::Result::UNSUPPORTED && pipeline == nullptr);
            }
        }
        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, pipeline));
        context.Track(pipeline);

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.format = color.format;
        textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
        textureDesc.width = width;
        textureDesc.height = height;
        textureDesc.layerNum = layerNum;
        nri::Texture* texture = nullptr;
        TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

        nri::TextureViewDesc viewDesc = {};
        viewDesc.texture = texture;
        viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
        viewDesc.format = color.format;
        viewDesc.mipNum = 1;
        viewDesc.layerNum = layerNum;
        viewDesc.sliceNum = 1;
        nri::Descriptor* view = nullptr;
        TEST_CHECK(context.core.CreateTextureView(viewDesc, view));
        context.Track(view);

        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = slicePitch * 2;
        nri::Buffer* readback = nullptr;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));

        nri::CommandAllocator* allocator = nullptr;
        nri::CommandBuffer* commands = nullptr;
        TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
        TEST_CHECK(context.core.BeginCommandBuffer(*commands, nullptr));
        nri::TextureBarrierDesc barrier = {};
        barrier.texture = texture;
        barrier.mipNum = 1;
        barrier.layerNum = layerNum;
        barrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
        nri::BarrierDesc barriers = {};
        barriers.textures = &barrier;
        barriers.textureNum = 1;
        context.core.CmdBarrier(*commands, barriers);

        nri::AttachmentDesc attachment = {};
        attachment.descriptor = view;
        attachment.loadOp = nri::LoadOp::CLEAR;
        attachment.storeOp = nri::StoreOp::STORE;
        nri::RenderingDesc rendering = {};
        rendering.colors = &attachment;
        rendering.colorNum = 1;
        rendering.viewMask = viewMasks[caseIndex];
        context.core.CmdBeginRendering(*commands, rendering);
        context.core.CmdSetPipelineLayout(*commands, nri::BindPoint::GRAPHICS, *pipelineLayout);
        context.core.CmdSetPipeline(*commands, *pipeline);
        const nri::Viewport viewport = {0, 0, width, height, 0, 1};
        const nri::Rect scissor = {0, 0, width, height};
        context.core.CmdSetViewports(*commands, &viewport, 1);
        context.core.CmdSetScissors(*commands, &scissor, 1);
        if (viewportBased || routing) {
            const nri::Viewport viewports[] = {{0, 0, width / 2, height, 0, 1}, {width / 2, 0, width / 2, height, 0, 1}};
            const nri::Rect scissors[] = {scissor, scissor};
            context.core.CmdSetViewports(*commands, viewports, 2);
            context.core.CmdSetScissors(*commands, scissors, 2);
        }
        context.core.CmdDraw(*commands, {3, routing ? 2u : 1u, 0, 0});
        context.core.CmdEndRendering(*commands);

        barrier.before = barrier.after;
        barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
        context.core.CmdBarrier(*commands, barriers);
        for (uint16_t layer = 0; layer < layerNum; layer++) {
            nri::TextureRegionDesc region = {};
            region.layerOffset = layer;
            region.width = width;
            region.height = height;
            region.depth = 1;
            const nri::TextureDataLayoutDesc dataLayout = {uint64_t(layer) * slicePitch, rowPitch, slicePitch};
            context.core.CmdReadbackTextureToBuffer(*commands, *readback, dataLayout, *texture, region);
        }
        TEST_CHECK(context.SubmitAndWait(queue, *commands));

        const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
        TEST_CHECK(data != nullptr);
        bool casePassed = true;
        for (uint32_t layer = 0; layer < layerNum; layer++) {
            // A sparse mask maps packed amplification index 0 to view/layer 1.
            const std::array<uint8_t, 4> expected = caseIndex == 1 || flexible ? (layer == 0 ? std::array<uint8_t, 4>{0, 0, 0, 0} : std::array<uint8_t, 4>{255, 0, 0, 255}) : (layer == 0 ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 255, 0, 255});
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    std::array<uint8_t, 4> pixel = viewportBased && x >= width / 2 ? std::array<uint8_t, 4>{0, 255, 0, 255} : expected;
                    if (routing)
                        pixel = layer == 0 ? (x >= width / 2 ? std::array<uint8_t, 4>{0, 255, 0, 255} : std::array<uint8_t, 4>{0, 0, 0, 0}) : (x < width / 2 ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 0, 0, 0});
                    for (uint32_t c = 0; c < 4; c++)
                        casePassed &= data[layer * slicePitch + y * rowPitch + x * 4 + c] == pixel[c];
                }
            }
        }
        if (!casePassed)
            printf("Multiview first pixels: layer 0 = %u,%u,%u,%u; layer 1 = %u,%u,%u,%u\n", data[0], data[1], data[2], data[3], data[slicePitch], data[slicePitch + 1], data[slicePitch + 2], data[slicePitch + 3]);
        context.core.UnmapBuffer(*readback);
        passed &= test::Report(names[caseIndex], casePassed);
    }

    return passed;
}

bool TestAttachmentClears(test::Context& context, nri::Queue& queue, nri::Format format, bool reinterpretFormat = false) {
    const bool depthStencil = format == nri::Format::D32_SFLOAT_S8_UINT;
    const bool signedInteger = format == nri::Format::R32_SINT;
    constexpr uint32_t width = 7, height = 5, slices = 3;
    constexpr uint32_t rowPitch = 256, slicePitch = rowPitch * height;
    nri::TextureDesc textureDesc = {};
    textureDesc.type = depthStencil ? nri::TextureType::TEXTURE_2D : nri::TextureType::TEXTURE_3D;
    textureDesc.format = reinterpretFormat ? nri::Format::R32_UINT : format;
    textureDesc.width = width * 2;
    textureDesc.height = height * 2;
    textureDesc.depth = depthStencil ? 1 : slices * 2;
    textureDesc.layerNum = depthStencil ? slices : 1;
    textureDesc.mipNum = 2;
    textureDesc.usage = depthStencil ? nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT : nri::TextureUsageBits::COLOR_ATTACHMENT;
    if (depthStencil)
        textureDesc.usage |= nri::TextureUsageBits::SHADER_RESOURCE;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = slices * slicePitch * (depthStencil ? 2 : 1);
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commands = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
    TEST_CHECK(context.core.BeginCommandBuffer(*commands, nullptr));
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = texture;
    barrier.after = depthStencil ? nri::AccessLayoutStage{nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_WRITE, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT} : nri::AccessLayoutStage{nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
    barrier.mipOffset = 1;
    barrier.mipNum = 1;
    barrier.layerNum = textureDesc.layerNum;
    nri::BarrierDesc barriers = {};
    barriers.textures = &barrier;
    barriers.textureNum = 1;
    context.core.CmdBarrier(*commands, barriers);
    for (uint32_t slice = 0; slice < slices; slice++) {
        nri::TextureViewDesc viewDesc = {};
        viewDesc.texture = texture;
        viewDesc.type = depthStencil ? nri::TextureView::DEPTH_STENCIL_ATTACHMENT : nri::TextureView::COLOR_ATTACHMENT;
        viewDesc.format = format;
        viewDesc.mipOffset = 1;
        viewDesc.mipNum = 1;
        viewDesc.layerOffset = depthStencil ? slice : 0;
        viewDesc.layerNum = 1;
        viewDesc.sliceOffset = depthStencil ? 0 : slice;
        viewDesc.sliceNum = 1;
        viewDesc.planes = depthStencil ? nri::PlaneBits::DEPTH | nri::PlaneBits::STENCIL : nri::PlaneBits::COLOR;
        nri::Descriptor* view = nullptr;
        TEST_CHECK(context.core.CreateTextureView(viewDesc, view));
        context.Track(view);
        nri::AttachmentDesc attachment = {};
        attachment.descriptor = view;
        attachment.loadOp = nri::LoadOp::CLEAR;
        attachment.storeOp = nri::StoreOp::STORE;
        if (depthStencil)
            attachment.clearValue.depthStencil = {0.125f * (slice + 1), uint8_t(17 + slice)};
        else if (signedInteger)
            attachment.clearValue.color.i.x = -100000003 - int32_t(slice);
        else
            attachment.clearValue.color.ui.x = 0x89ABCDEF + slice;
        nri::RenderingDesc rendering = {};
        if (depthStencil)
            rendering.depth = attachment;
        else {
            rendering.colors = &attachment;
            rendering.colorNum = 1;
        }
        context.core.CmdBeginRendering(*commands, rendering);
        if (slice == 1) {
            nri::ClearAttachmentDesc clear = {};
            clear.planes = viewDesc.planes;
            if (depthStencil)
                clear.value.depthStencil = {0.75f, 213};
            else if (signedInteger)
                clear.value.color.i.x = -123456789;
            else
                clear.value.color.ui.x = 0xFEDCBA98;
            const nri::Rect rect = {2, 1, 3, 2};
            context.core.CmdClearAttachments(*commands, &clear, 1, &rect, 1);
        }
        context.core.CmdEndRendering(*commands);
    }
    barrier.before = barrier.after;
    barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    context.core.CmdBarrier(*commands, barriers);
    for (uint32_t slice = 0; slice < slices; slice++) {
        nri::TextureRegionDesc region = {};
        region.width = width;
        region.height = height;
        region.depth = 1;
        region.z = depthStencil ? 0 : slice;
        region.layerOffset = depthStencil ? slice : 0;
        region.mipOffset = 1;
        region.planes = depthStencil ? nri::PlaneBits::DEPTH : nri::PlaneBits::COLOR;
        nri::TextureDataLayoutDesc layout = {slice * slicePitch, rowPitch, slicePitch};
        context.core.CmdReadbackTextureToBuffer(*commands, *readback, layout, *texture, region);
        if (depthStencil) {
            region.planes = nri::PlaneBits::STENCIL;
            layout.offset += slices * slicePitch;
            context.core.CmdReadbackTextureToBuffer(*commands, *readback, layout, *texture, region);
        }
    }
    TEST_CHECK(context.SubmitAndWait(queue, *commands));
    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t slice = 0; slice < slices; slice++) {
        for (uint32_t y = 0; y < height; y++) {
            const uint32_t* row = (const uint32_t*)(data + slice * slicePitch + y * rowPitch);
            for (uint32_t x = 0; x < width; x++) {
                const bool inside = slice == 1 && x >= 2 && x < 5 && y >= 1 && y < 3;
                if (depthStencil) {
                    float value;
                    memcpy(&value, row + x, sizeof(value));
                    passed &= value == (inside ? 0.75f : 0.125f * (slice + 1));
                    passed &= data[(slices + slice) * slicePitch + y * rowPitch + x] == (inside ? 213 : 17 + slice);
                } else {
                    const uint32_t expected = signedInteger ? uint32_t(inside ? -123456789 : -100000003 - int32_t(slice)) : (inside ? 0xFEDCBA98 : 0x89ABCDEF + slice);
                    passed &= row[x] == expected;
                }
            }
        }
    }
    context.core.UnmapBuffer(*readback);

    if (depthStencil) {
        nri::Descriptor* views[2] = {};
        nri::TextureViewDesc viewDesc = {};
        viewDesc.texture = texture;
        viewDesc.type = nri::TextureView::TEXTURE;
        viewDesc.format = format;
        viewDesc.mipOffset = 1;
        viewDesc.mipNum = 1;
        viewDesc.layerOffset = 1;
        viewDesc.layerNum = 1;
        for (uint32_t i = 0; i < 2; i++) {
            viewDesc.planes = i ? nri::PlaneBits::STENCIL : nri::PlaneBits::DEPTH;
            TEST_CHECK(context.core.CreateTextureView(viewDesc, views[i]));
            context.Track(views[i]);
        }
        const nri::DescriptorRangeDesc range = {0, 2, nri::DescriptorType::TEXTURE, nri::StageBits::COMPUTE_SHADER};
        nri::DescriptorSetDesc setDesc = {};
        setDesc.registerSpace = 1;
        setDesc.ranges = &range;
        setDesc.rangeNum = 1;
        const nri::RootDescriptorDesc root = {0, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER};
        nri::PipelineLayoutDesc layoutDesc = {};
        layoutDesc.descriptorSets = &setDesc;
        layoutDesc.descriptorSetNum = 1;
        layoutDesc.rootDescriptors = &root;
        layoutDesc.rootDescriptorNum = 1;
        layoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        nri::PipelineLayout* layout = nullptr;
        TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, layout));
        context.Track(layout);
        nri::ComputePipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = layout;
        pipelineDesc.shader = LoadComputeShader(context, "MetalTests.metallib", "readDepthStencil");
        pipelineDesc.shader.threadGroupSizeX = 8;
        pipelineDesc.shader.threadGroupSizeY = 8;
        pipelineDesc.shader.threadGroupSizeZ = 1;
        nri::Pipeline* pipeline = nullptr;
        TEST_CHECK(context.core.CreateComputePipeline(*context.device, pipelineDesc, pipeline));
        context.Track(pipeline);
        nri::DescriptorPoolDesc poolDesc = {};
        poolDesc.descriptorSetMaxNum = 1;
        poolDesc.textureMaxNum = 2;
        nri::DescriptorPool* pool = nullptr;
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, pool));
        context.Track(pool);
        nri::DescriptorSet* set = nullptr;
        TEST_CHECK(context.core.AllocateDescriptorSets(*pool, *layout, 0, &set, 1, 0));
        const nri::UpdateDescriptorRangeDesc update = {set, 0, 0, views, 2};
        context.core.UpdateDescriptorRanges(&update, 1);
        bufferDesc.size = width * height * 8;
        bufferDesc.structureStride = 8;
        bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;
        nri::Buffer* output = nullptr;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, output));
        nri::BufferViewDesc outputDesc = {};
        outputDesc.buffer = output;
        outputDesc.type = nri::BufferView::STORAGE_STRUCTURED_BUFFER;
        outputDesc.size = nri::WHOLE_SIZE;
        nri::Descriptor* outputView = nullptr;
        TEST_CHECK(context.core.CreateBufferView(outputDesc, outputView));
        context.Track(outputView);
        TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
        TEST_CHECK(context.core.BeginCommandBuffer(*commands, pool));
        barrier.before = barrier.after;
        barrier.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};
        context.core.CmdBarrier(*commands, barriers);
        context.core.CmdSetPipelineLayout(*commands, nri::BindPoint::COMPUTE, *layout);
        context.core.CmdSetPipeline(*commands, *pipeline);
        context.core.CmdSetDescriptorSet(*commands, {0, set, nri::BindPoint::COMPUTE});
        context.core.CmdSetRootDescriptor(*commands, {0, outputView, 0, nri::BindPoint::COMPUTE});
        context.core.CmdDispatch(*commands, {1, 1, 1});
        nri::BufferBarrierDesc outputBarrier = {};
        outputBarrier.buffer = output;
        outputBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        outputBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
        barriers = {};
        barriers.buffers = &outputBarrier;
        barriers.bufferNum = 1;
        context.core.CmdBarrier(*commands, barriers);
        context.core.CmdCopyBuffer(*commands, *readback, 0, *output, 0, bufferDesc.size);
        TEST_CHECK(context.SubmitAndWait(queue, *commands));
        const uint32_t* sampled = (const uint32_t*)context.core.MapBuffer(*readback, 0, bufferDesc.size);
        TEST_CHECK(sampled != nullptr);
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                const bool inside = x >= 2 && x < 5 && y >= 1 && y < 3;
                float depth;
                memcpy(&depth, sampled + (y * width + x) * 2, sizeof(depth));
                passed &= depth == (inside ? 0.75f : 0.25f);
                passed &= sampled[(y * width + x) * 2 + 1] == (inside ? 213 : 18);
            }
        }
        context.core.UnmapBuffer(*readback);
        TEST_CHECK(test::Report("sampled depth and stencil views at nonzero mip/layer", passed));
    }

    return test::Report(reinterpretFormat ? "signed attachment view of unsigned 3D texture, mip/slice clears" : (depthStencil ? "depth/stencil load and rectangle clears, separate plane readback" : (signedInteger ? "signed integer load/rectangle clears on 3D mip slices" : "unsigned integer load/rectangle clears on 3D mip slices")), passed);
}

bool TestNativeRayDispatch(test::Context& context, nri::Queue& queue) {
    if (!context.deviceDesc->tiers.rayTracing) {
        printf("SKIP  native ray dispatch requires Metal 4 ray tracing support\n");

        return true;
    }

    nri::RayTracingInterface ray = {};
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::RayTracingInterface), &ray));
    const nri::RootConstantDesc constantDesc = {0, sizeof(Constants), nri::StageBits::RAYGEN_SHADER};
    const nri::RootDescriptorDesc rootDescs[] = {
        {0, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::RAYGEN_SHADER},
        {0, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::RAYGEN_SHADER},
    };
    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.rootConstants = &constantDesc;
    layoutDesc.rootConstantNum = 1;
    layoutDesc.rootDescriptors = rootDescs;
    layoutDesc.rootDescriptorNum = 2;
    layoutDesc.shaderStages = nri::StageBits::RAYGEN_SHADER;
    nri::PipelineLayout* layout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, layout));
    context.Track(layout);
    nri::ShaderDesc shaders[] = {
        LoadComputeShader(context, "MetalTests.metallib", "nativeRaygen"),
        LoadComputeShader(context, "MetalTests.metallib", "nativeRaygenAlternate"),
    };
    for (nri::ShaderDesc& shader : shaders) {
        TEST_CHECK(shader.bytecode != nullptr);
        shader.stage = nri::StageBits::RAYGEN_SHADER;
    }
    const nri::ShaderLibraryDesc library = {shaders, 2};
    const nri::ShaderGroupDesc groups[] = {{{1, 0, 0}}, {{2, 0, 0}}};
    nri::RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = layout;
    pipelineDesc.shaderLibrary = &library;
    pipelineDesc.shaderGroups = groups;
    pipelineDesc.shaderGroupNum = 2;
    pipelineDesc.recursionMaxDepth = 1;
    pipelineDesc.flags = nri::RayTracingPipelineBits::SKIP_TRIANGLES | nri::RayTracingPipelineBits::SKIP_AABBS;
    nri::Pipeline* pipeline = nullptr;
    if (context.deviceDesc->features.shaderBytecodeDXIL) {
        nri::ShaderDesc mixedShaders[] = {shaders[0], LoadComputeShader(context, "RayTracingTriangle.rgen.dxil")};
        mixedShaders[1].stage = nri::StageBits::RAYGEN_SHADER;
        TEST_CHECK(mixedShaders[1].bytecode != nullptr);
        const nri::ShaderLibraryDesc mixedLibrary = {mixedShaders, 2};
        nri::RayTracingPipelineDesc mixedDesc = pipelineDesc;
        mixedDesc.shaderLibrary = &mixedLibrary;
        TEST_CHECK(ray.CreateRayTracingPipeline(*context.device, mixedDesc, pipeline) == nri::Result::UNSUPPORTED && pipeline == nullptr);
    }
    TEST_CHECK(ray.CreateRayTracingPipeline(*context.device, pipelineDesc, pipeline));
    context.Track(pipeline);

    constexpr uint32_t width = 7, height = 3, depth = 2;
    constexpr uint32_t count = width * height * depth;
    const Constants constants = {count, 13, 71, 0x319A5};
    const uint64_t offset = std::max<uint64_t>(64, context.deviceDesc->memoryAlignment.bufferShaderResourceOffset);
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = offset + (count + GUARD_NUM) * sizeof(uint32_t);
    bufferDesc.structureStride = sizeof(uint32_t);
    bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;
    nri::Buffer* output = nullptr;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, output));
    bufferDesc.usage = nri::BufferUsageBits::NONE;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));
    std::vector<uint32_t> initial(bufferDesc.size / sizeof(uint32_t), SENTINEL);
    nri::BufferViewDesc viewDesc = {};
    viewDesc.buffer = output;
    viewDesc.type = nri::BufferView::STORAGE_STRUCTURED_BUFFER;
    viewDesc.offset = offset;
    viewDesc.size = count * sizeof(uint32_t);
    nri::Descriptor* view = nullptr;
    TEST_CHECK(context.core.CreateBufferView(viewDesc, view));
    context.Track(view);

    const uint64_t recordStride = context.deviceDesc->memoryAlignment.shaderBindingTable;
    bufferDesc.size = recordStride * 3;
    bufferDesc.usage = nri::BufferUsageBits::SHADER_BINDING_TABLE;
    bufferDesc.structureStride = 0;
    nri::Buffer* records = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, records));
    uint8_t* recordData = (uint8_t*)context.core.MapBuffer(*records, 0, nri::WHOLE_SIZE);
    TEST_CHECK(recordData != nullptr);
    memset(recordData, 0, bufferDesc.size);
    TEST_CHECK(ray.WriteShaderGroupIdentifiers(*pipeline, 0, 2, (uint32_t)recordStride, recordData + recordStride));
    context.core.UnmapBuffer(*records);

    for (uint32_t mode = 0; mode < 2; mode++) {
        const nri::BufferUploadDesc upload = {initial.data(), output, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::RAYGEN_SHADER}};
        TEST_CHECK(context.helper.UploadData(queue, nullptr, 0, &upload, 1));
        nri::CommandAllocator* allocator = nullptr;
        nri::CommandBuffer* commands = nullptr;
        TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
        TEST_CHECK(context.core.BeginCommandBuffer(*commands, nullptr));
        context.core.CmdSetPipelineLayout(*commands, nri::BindPoint::RAY_TRACING, *layout);
        context.core.CmdSetPipeline(*commands, *pipeline);
        context.core.CmdSetRootConstants(*commands, {0, &constants, sizeof(constants), 0, nri::BindPoint::RAY_TRACING});
        context.core.CmdSetRootDescriptor(*commands, {1, view, 0, nri::BindPoint::RAY_TRACING});
        if (mode == 0) {
            nri::DispatchRaysDesc dispatch = {};
            dispatch.raygenShaderRecord = {records, recordStride, 32, 0};
            dispatch.width = width;
            dispatch.height = height;
            dispatch.depth = depth;
            ray.CmdDispatchRays(*commands, dispatch);
        } else {
            bufferDesc.size = 32 + sizeof(nri::DispatchRaysIndirectDesc);
            bufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
            nri::Buffer* indirect = nullptr;
            TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, indirect));
            nri::DispatchRaysIndirectDesc dispatch = {};
            dispatch.raygenShaderRecordAddress = context.core.GetBufferDeviceAddress(*records) + recordStride * 2;
            dispatch.raygenShaderRecordSize = 32;
            dispatch.width = width;
            dispatch.height = height;
            dispatch.depth = depth;
            void* mapped = context.core.MapBuffer(*indirect, 0, nri::WHOLE_SIZE);
            TEST_CHECK(mapped != nullptr);
            memcpy((uint8_t*)mapped + 32, &dispatch, sizeof(dispatch));
            context.core.UnmapBuffer(*indirect);
            ray.CmdDispatchRaysIndirect(*commands, *indirect, 32);
        }
        nri::BufferBarrierDesc barrier = {};
        barrier.buffer = output;
        barrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::RAYGEN_SHADER};
        barrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
        nri::BarrierDesc barriers = {};
        barriers.buffers = &barrier;
        barriers.bufferNum = 1;
        context.core.CmdBarrier(*commands, barriers);
        context.core.CmdCopyBuffer(*commands, *readback, 0, *output, 0, nri::WHOLE_SIZE);
        TEST_CHECK(context.SubmitAndWait(queue, *commands));
        const uint32_t* result = (const uint32_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
        TEST_CHECK(result != nullptr);
        bool passed = true;
        for (uint32_t i = 0; i < initial.size(); i++) {
            uint32_t expected = SENTINEL;
            if (i >= offset / 4 && i < offset / 4 + count) {
                expected = uint32_t(i - offset / 4) * 13 + 71;
                if (mode)
                    expected ^= 0x319A5;
            }
            passed &= result[i] == expected;
        }
        context.core.UnmapBuffer(*readback);
        TEST_CHECK(test::Report(mode ? "native indirect ray dispatch selects alternate SBT record" : "native ray dispatch, 7x3x2 grid and guarded output", passed));
    }

    return true;
}

bool TestWrapping(const test::Settings& settings) {
    test::Context original;
    TEST_CHECK(original.Initialize(settings));
    nri::Queue* originalQueue = nullptr;
    TEST_CHECK(original.core.GetQueue(*original.device, nri::QueueType::GRAPHICS, 0, originalQueue));
    nri::DeviceCreationMetalDesc deviceDesc = {};
    deviceDesc.mtlDevice = original.core.GetDeviceNativeObject(original.device);
    deviceDesc.mtl4Queues[0] = original.core.GetQueueNativeObject(originalQueue);
    deviceDesc.enableNRIValidation = settings.debugNRI;
    test::Context wrapped;
    TEST_CHECK(nri::nriCreateDeviceFromMetalDevice(deviceDesc, wrapped.device));
    TEST_CHECK(nri::nriGetInterface(*wrapped.device, NRI_INTERFACE(nri::CoreInterface), &wrapped.core));
    nri::WrapperMetalInterface wrapper = {};
    TEST_CHECK(nri::nriGetInterface(*wrapped.device, NRI_INTERFACE(nri::WrapperMetalInterface), &wrapper));
    nri::Queue* queue = nullptr;
    TEST_CHECK(wrapped.core.GetQueue(*wrapped.device, nri::QueueType::GRAPHICS, 0, queue));
    TEST_CHECK(wrapped.core.GetQueueNativeObject(queue) == deviceDesc.mtl4Queues[0]);

#if METAL_NATIVE_TESTS
    MTL::SharedEvent* event = ((MTL::Device*)deviceDesc.mtlDevice)->newSharedEvent();
    TEST_CHECK(event != nullptr);
    event->setSignaledValue(7);
    const nri::FenceMetalDesc fenceDesc = {event};
    nri::Fence* fence = nullptr;
    const nri::Result fenceResult = wrapper.CreateFenceMetal(*wrapped.device, fenceDesc, fence);
    event->release();
    TEST_CHECK(fenceResult);
    wrapped.Track(fence);
    TEST_CHECK(wrapped.core.GetFenceValue(*fence) == 7);
    const nri::FenceSubmitDesc waitFence = {fence, 7, nri::StageBits::ALL};
    const nri::FenceSubmitDesc signalFence = {fence, 19, nri::StageBits::ALL};
#endif

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = 768;
    nri::Buffer* originalBuffer = nullptr;
    TEST_CHECK(original.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, originalBuffer));
    const nri::BufferMetalDesc wrappedBufferDesc = {(void*)original.core.GetBufferNativeObject(originalBuffer), bufferDesc};
    nri::Buffer* upload = nullptr;
    TEST_CHECK(wrapper.CreateBufferMetal(*wrapped.device, wrappedBufferDesc, upload));
    wrapped.Track(upload);
    nri::Buffer* duplicateBuffer = nullptr;
    TEST_CHECK(wrapper.CreateBufferMetal(*wrapped.device, wrappedBufferDesc, duplicateBuffer));
    wrapped.core.DestroyBuffer(duplicateBuffer);
    original.core.DestroyBuffer(originalBuffer);
    original.buffers.clear();

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
    textureDesc.width = 4;
    textureDesc.height = 3;
    nri::Texture* originalTexture = nullptr;
    TEST_CHECK(original.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, originalTexture));
    const nri::TextureMetalDesc wrappedTextureDesc = {(void*)original.core.GetTextureNativeObject(originalTexture), textureDesc};
    nri::Texture* texture = nullptr;
    TEST_CHECK(wrapper.CreateTextureMetal(*wrapped.device, wrappedTextureDesc, texture));
    wrapped.Track(texture);
    nri::Texture* duplicateTexture = nullptr;
    TEST_CHECK(wrapper.CreateTextureMetal(*wrapped.device, wrappedTextureDesc, duplicateTexture));
    wrapped.core.DestroyTexture(duplicateTexture);
    original.core.DestroyTexture(originalTexture);
    original.textures.clear();

    std::array<uint8_t, 768> expected = {};
    for (uint32_t i = 0; i < expected.size(); i++)
        expected[i] = uint8_t(i * 37 + i / 256 * 11 + 3);
    void* mapped = wrapped.core.MapBuffer(*upload, 0, nri::WHOLE_SIZE);
    TEST_CHECK(mapped != nullptr);
    memcpy(mapped, expected.data(), expected.size());
    wrapped.core.UnmapBuffer(*upload);
    nri::Buffer* readback = nullptr;
    TEST_CHECK(wrapped.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(wrapped.CreateCommandObjects(*queue, allocator, commandBuffer));
    TEST_CHECK(wrapped.core.BeginCommandBuffer(*commandBuffer, nullptr));
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = texture;
    barrier.after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
    barrier.mipNum = 1;
    barrier.layerNum = 1;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &barrier;
    barrierDesc.textureNum = 1;
    wrapped.core.CmdBarrier(*commandBuffer, barrierDesc);
    nri::TextureRegionDesc region = {};
    region.width = 4;
    region.height = 3;
    region.depth = 1;
    const nri::TextureDataLayoutDesc dataLayout = {0, 256, 768};
    wrapped.core.CmdUploadBufferToTexture(*commandBuffer, *texture, region, *upload, dataLayout);
    barrier.before = barrier.after;
    barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    wrapped.core.CmdBarrier(*commandBuffer, barrierDesc);
    wrapped.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, dataLayout, *texture, region);
#if METAL_NATIVE_TESTS
    TEST_CHECK(wrapped.core.EndCommandBuffer(*commandBuffer));
    nri::QueueSubmitDesc submit = {};
    submit.commandBuffers = &commandBuffer;
    submit.commandBufferNum = 1;
    submit.waitFences = &waitFence;
    submit.waitFenceNum = 1;
    submit.signalFences = &signalFence;
    submit.signalFenceNum = 1;
    TEST_CHECK(wrapped.core.QueueSubmit(*queue, submit));
    wrapped.core.Wait(*fence, 19);
    TEST_CHECK(wrapped.core.GetFenceValue(*fence) == 19);
    TEST_CHECK(wrapped.core.QueueWaitIdle(queue));
    TEST_CHECK(test::Report("wrapped shared event retains ownership and signals GPU completion", true));
#else
    TEST_CHECK(wrapped.SubmitAndWait(*queue, *commandBuffer));
#endif
    const uint8_t* data = (const uint8_t*)wrapped.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t y = 0; y < 3; y++)
        passed &= memcmp(data + y * 256, expected.data() + y * 256, 16) == 0;
    wrapped.core.UnmapBuffer(*readback);

    return test::Report("wrapped device, queue, buffer and texture retain ownership", passed);
}

bool TestSamplerBorderColors(test::Context& context) {
    nri::SamplerDesc desc = {};
    desc.addressModes.u = nri::AddressMode::CLAMP_TO_BORDER;
    nri::Descriptor* sampler = nullptr;

    const nri::Color acceptedFloatColors[] = {{{0.0f, 0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f, 1.0f}}, {{1.0f, 1.0f, 1.0f, 1.0f}}};
    for (const nri::Color& color : acceptedFloatColors) {
        desc.borderColor = color;
        TEST_CHECK(context.core.CreateSampler(*context.device, desc, sampler));
        context.core.DestroyDescriptor(sampler);
        sampler = nullptr;
    }

    desc.isInteger = true;
    const nri::Color32ui acceptedIntegerColors[] = {{0, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 1, 1}};
    for (const nri::Color32ui& color : acceptedIntegerColors) {
        desc.borderColor.ui = color;
        TEST_CHECK(context.core.CreateSampler(*context.device, desc, sampler));
        context.core.DestroyDescriptor(sampler);
        sampler = nullptr;
    }

    desc.borderColor.ui = {1, 0, 1, 1};
    TEST_CHECK(context.core.CreateSampler(*context.device, desc, sampler) == nri::Result::UNSUPPORTED && sampler == nullptr);
    desc.isInteger = false;
    desc.borderColor.f = {1.0f, 0.25f, 1.0f, 1.0f};
    TEST_CHECK(context.core.CreateSampler(*context.device, desc, sampler) == nri::Result::UNSUPPORTED && sampler == nullptr);

    desc.addressModes.u = nri::AddressMode::REPEAT;
    TEST_CHECK(context.core.CreateSampler(*context.device, desc, sampler));
    context.core.DestroyDescriptor(sampler);

    return test::Report("Metal sampler border color acceptance boundary", true);
}

bool TestDynamicVertexStrides(test::Context& context, nri::Queue& queue) {
    constexpr uint32_t vertexStride = 24;
    constexpr uint32_t instanceStride = 40;
    std::array<uint8_t, vertexStride * 3 + 8 * 3> vertices = {};
    const float positions[][2] = {{-0.5f, -1.0f}, {0.5f, -1.0f}, {-0.5f, 3.0f}};
    for (uint32_t i = 0; i < 3; i++)
        memcpy(vertices.data() + i * vertexStride, positions[i], sizeof(positions[i]));
    memcpy(vertices.data() + vertexStride * 3, positions, sizeof(positions));
    std::array<uint8_t, instanceStride * 2> instances = {};
    const float offsets[][2] = {{-0.5f, 0.0f}, {0.5f, 0.0f}};
    const uint32_t colors[] = {0xFF0000FF, 0xFF00FF00};
    for (uint32_t i = 0; i < 2; i++) {
        memcpy(instances.data() + i * instanceStride, offsets[i], sizeof(offsets[i]));
        memcpy(instances.data() + i * instanceStride + 8, colors + i, sizeof(colors[i]));
    }

    nri::BufferDesc bufferDesc = {};
    bufferDesc.usage = nri::BufferUsageBits::VERTEX;
    bufferDesc.size = vertices.size();
    nri::Buffer* vertexBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, vertexBuffer));
    bufferDesc.size = instances.size();
    nri::Buffer* instanceBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, instanceBuffer));
    const nri::BufferUploadDesc uploads[] = {
        {vertices.data(), vertexBuffer, {nri::AccessBits::VERTEX_BUFFER, nri::StageBits::VERTEX_SHADER}},
        {instances.data(), instanceBuffer, {nri::AccessBits::VERTEX_BUFFER, nri::StageBits::VERTEX_SHADER}},
    };
    TEST_CHECK(context.helper.UploadData(queue, nullptr, 0, uploads, 2));

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* layout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, layout));
    context.Track(layout);
    const nri::VertexStreamDesc streams[] = {{0, nri::VertexStreamStepRate::PER_VERTEX, 0}, {1, nri::VertexStreamStepRate::PER_INSTANCE, 0}};
    const nri::VertexAttributeDesc attributes[] = {
        {{"POSITION", 0}, {0}, 0, nri::Format::RG32_SFLOAT, 0},
        {{"TEXCOORD", 0}, {1}, 0, nri::Format::RG32_SFLOAT, 1},
        {{"COLOR", 0}, {2}, 8, nri::Format::RGBA8_UNORM, 1},
    };
    const nri::VertexInputDesc vertexInput = {attributes, 3, streams, 2};
    nri::ColorAttachmentDesc color = {};
    color.format = nri::Format::RGBA8_UNORM;
    color.colorWriteMask = nri::ColorWriteBits::RGBA;

    bool passed = true;
    const uint32_t shaderKindNum = context.deviceDesc->features.shaderBytecodeDXIL ? 2 : 1;
    for (uint32_t shaderKind = 0; shaderKind < shaderKindNum; shaderKind++) {
        nri::ShaderDesc shaders[] = {
            LoadComputeShader(context, shaderKind ? "MetalVertexStride.vs.dxil" : "MetalTests.metallib", shaderKind ? nullptr : "vertexStrideVertex"),
            LoadComputeShader(context, shaderKind ? "MetalVertexStride.fs.dxil" : "MetalTests.metallib", shaderKind ? nullptr : "vertexStrideFragment"),
        };
        shaders[0].stage = nri::StageBits::VERTEX_SHADER;
        shaders[1].stage = nri::StageBits::FRAGMENT_SHADER;
        nri::GraphicsPipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = layout;
        pipelineDesc.vertexInput = &vertexInput;
        pipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
        pipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
        pipelineDesc.outputMerger.colors = &color;
        pipelineDesc.outputMerger.colorNum = 1;
        pipelineDesc.shaders = shaders;
        pipelineDesc.shaderNum = 2;
        nri::Pipeline* pipeline = nullptr;
        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, pipeline));
        context.Track(pipeline);

        nri::TextureDesc targetDesc = {};
        targetDesc.type = nri::TextureType::TEXTURE_2D;
        targetDesc.format = color.format;
        targetDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
        targetDesc.width = 4;
        targetDesc.height = 1;
        nri::Texture* target = nullptr;
        TEST_CHECK(context.CreateTexture(targetDesc, nri::MemoryLocation::DEVICE, target));
        nri::TextureViewDesc viewDesc = {};
        viewDesc.texture = target;
        viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
        viewDesc.format = color.format;
        viewDesc.mipNum = 1;
        viewDesc.layerNum = 1;
        viewDesc.sliceNum = 1;
        nri::Descriptor* targetView = nullptr;
        TEST_CHECK(context.core.CreateTextureView(viewDesc, targetView));
        context.Track(targetView);
        bufferDesc = {};
        bufferDesc.size = 256;
        nri::Buffer* readback = nullptr;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));

        nri::CommandAllocator* allocator = nullptr;
        nri::CommandBuffer* commands = nullptr;
        TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
        TEST_CHECK(context.core.BeginCommandBuffer(*commands, nullptr));
        nri::TextureBarrierDesc barrier = {};
        barrier.texture = target;
        barrier.mipNum = 1;
        barrier.layerNum = 1;
        barrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
        nri::BarrierDesc barriers = {};
        barriers.textures = &barrier;
        barriers.textureNum = 1;
        context.core.CmdBarrier(*commands, barriers);
        nri::AttachmentDesc attachment = {};
        attachment.descriptor = targetView;
        attachment.loadOp = nri::LoadOp::CLEAR;
        attachment.storeOp = nri::StoreOp::STORE;
        nri::RenderingDesc rendering = {};
        rendering.colors = &attachment;
        rendering.colorNum = 1;
        context.core.CmdBeginRendering(*commands, rendering);
        context.core.CmdSetPipelineLayout(*commands, nri::BindPoint::GRAPHICS, *layout);
        context.core.CmdSetPipeline(*commands, *pipeline);
        const nri::VertexBufferDesc vertexBinding = {vertexBuffer, 0, vertexStride};
        const nri::VertexBufferDesc instanceBinding = {instanceBuffer, 0, instanceStride};
        context.core.CmdSetVertexBuffers(*commands, 0, &vertexBinding, 1);
        context.core.CmdSetVertexBuffers(*commands, 1, &instanceBinding, 1);
        const nri::Viewport viewport = {0, 0, 2, 1, 0, 1};
        const nri::Rect scissor = {0, 0, 4, 1};
        context.core.CmdSetViewports(*commands, &viewport, 1);
        context.core.CmdSetScissors(*commands, &scissor, 1);
        context.core.CmdDraw(*commands, {3, 2, 0, 0});
        const nri::VertexBufferDesc packedBinding = {vertexBuffer, vertexStride * 3, 8};
        context.core.CmdSetVertexBuffers(*commands, 0, &packedBinding, 1);
        const nri::Viewport secondViewport = {2, 0, 2, 1, 0, 1};
        context.core.CmdSetViewports(*commands, &secondViewport, 1);
        context.core.CmdDraw(*commands, {3, 2, 0, 0});
        context.core.CmdEndRendering(*commands);
        barrier.before = barrier.after;
        barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
        context.core.CmdBarrier(*commands, barriers);
        const nri::TextureRegionDesc region = {0, 0, 0, 4, 1, 1, 0, 0};
        const nri::TextureDataLayoutDesc dataLayout = {0, 256, 256};
        context.core.CmdReadbackTextureToBuffer(*commands, *readback, dataLayout, *target, region);
        TEST_CHECK(context.SubmitAndWait(queue, *commands));
        const uint32_t* data = (const uint32_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
        TEST_CHECK(data != nullptr);
        const bool casePassed = data[0] == colors[0] && data[1] == colors[1] && data[2] == colors[0] && data[3] == colors[1];
        context.core.UnmapBuffer(*readback);
        passed &= test::Report(shaderKind ? "converted stage-in dynamic asymmetric vertex strides" : "native stage-in dynamic asymmetric vertex strides", casePassed);
    }

    return passed;
}

bool TestSamplerLodBias(test::Context& context, nri::Queue& queue) {
    constexpr uint16_t textureSize = 4;
    constexpr uint32_t rowPitch = 256;
    const std::array<std::array<uint8_t, 4>, 3> mipColors = {{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}}};
    std::array<std::vector<uint8_t>, 3> mipData;
    std::array<nri::TextureSubresourceUploadDesc, 3> subresources = {};
    for (uint32_t mip = 0; mip < 3; mip++) {
        const uint32_t size = textureSize >> mip;
        mipData[mip].resize(size * size * 4);
        for (size_t i = 0; i < mipData[mip].size(); i += 4)
            memcpy(mipData[mip].data() + i, mipColors[mip].data(), 4);
        subresources[mip] = {mipData[mip].data(), 1, size * 4, size * size * 4};
    }

    nri::TextureDesc sourceDesc = {};
    sourceDesc.type = nri::TextureType::TEXTURE_2D;
    sourceDesc.format = nri::Format::RGBA8_UNORM;
    sourceDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
    sourceDesc.width = textureSize;
    sourceDesc.height = textureSize;
    sourceDesc.mipNum = 3;
    nri::Texture* source = nullptr;
    TEST_CHECK(context.CreateTexture(sourceDesc, nri::MemoryLocation::DEVICE, source));
    const nri::TextureUploadDesc upload = {subresources.data(), source, {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::FRAGMENT_SHADER}, nri::PlaneBits::COLOR};
    TEST_CHECK(context.helper.UploadData(queue, &upload, 1, nullptr, 0));

    nri::TextureViewDesc sourceViewDesc = {};
    sourceViewDesc.texture = source;
    sourceViewDesc.type = nri::TextureView::TEXTURE;
    sourceViewDesc.format = sourceDesc.format;
    sourceViewDesc.mipNum = 3;
    sourceViewDesc.layerNum = 1;
    sourceViewDesc.sliceNum = 1;
    nri::Descriptor* sourceView = nullptr;
    TEST_CHECK(context.core.CreateTextureView(sourceViewDesc, sourceView));
    context.Track(sourceView);

    const nri::DescriptorRangeDesc ranges[] = {
        {0, 1, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER},
        {0, 1, nri::DescriptorType::SAMPLER, nri::StageBits::FRAGMENT_SHADER},
    };
    nri::DescriptorSetDesc setDesc = {};
    setDesc.ranges = ranges;
    setDesc.rangeNum = 2;
    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = &setDesc;
    layoutDesc.descriptorSetNum = 1;
    layoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* layout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, layout));
    context.Track(layout);

    nri::DescriptorPoolDesc poolDesc = {};
    poolDesc.descriptorSetMaxNum = 4;
    poolDesc.textureMaxNum = 4;
    poolDesc.samplerMaxNum = 4;
    nri::DescriptorPool* pool = nullptr;
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, pool));
    context.Track(pool);

    nri::ColorAttachmentDesc color = {};
    color.format = nri::Format::RGBA8_UNORM;
    color.colorWriteMask = nri::ColorWriteBits::RGBA;
    bool passed = true;
    const bool convertedSupported = context.deviceDesc->features.shaderBytecodeDXIL;
    for (uint32_t shaderKind = 0; shaderKind < (convertedSupported ? 2u : 1u); shaderKind++) {
        nri::ShaderDesc shaders[] = {
            LoadComputeShader(context, shaderKind ? "MetalSamplerLodBias.vs.dxil" : "MetalTests.metallib", shaderKind ? nullptr : "samplerLodBiasVertex"),
            LoadComputeShader(context, shaderKind ? "MetalSamplerLodBias.fs.dxil" : "MetalTests.metallib", shaderKind ? nullptr : "samplerLodBiasFragment"),
        };
        shaders[0].stage = nri::StageBits::VERTEX_SHADER;
        shaders[1].stage = nri::StageBits::FRAGMENT_SHADER;
        nri::GraphicsPipelineDesc pipelineDesc = {};
        pipelineDesc.pipelineLayout = layout;
        pipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
        pipelineDesc.outputMerger.colors = &color;
        pipelineDesc.outputMerger.colorNum = 1;
        pipelineDesc.shaders = shaders;
        pipelineDesc.shaderNum = 2;
        nri::Pipeline* pipeline = nullptr;
        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, pipeline));
        context.Track(pipeline);

        for (uint32_t biasIndex = 0; biasIndex < 2; biasIndex++) {
            const float bias = biasIndex ? -1.0f : 1.0f;
            const uint16_t outputSize = biasIndex ? 2 : 4;
            nri::SamplerDesc samplerDesc = {};
            samplerDesc.mipBias = bias;
            samplerDesc.mipMax = 2.0f;
            nri::Descriptor* sampler = nullptr;
            TEST_CHECK(context.core.CreateSampler(*context.device, samplerDesc, sampler));
            context.Track(sampler);
            nri::DescriptorSet* set = nullptr;
            TEST_CHECK(context.core.AllocateDescriptorSets(*pool, *layout, 0, &set, 1, 0));
            const nri::UpdateDescriptorRangeDesc updates[] = {{set, 0, 0, &sourceView, 1}, {set, 1, 0, &sampler, 1}};
            context.core.UpdateDescriptorRanges(updates, 2);

            nri::TextureDesc targetDesc = sourceDesc;
            targetDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
            targetDesc.width = outputSize;
            targetDesc.height = outputSize;
            targetDesc.mipNum = 1;
            nri::Texture* target = nullptr;
            TEST_CHECK(context.CreateTexture(targetDesc, nri::MemoryLocation::DEVICE, target));
            nri::TextureViewDesc targetViewDesc = sourceViewDesc;
            targetViewDesc.texture = target;
            targetViewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
            targetViewDesc.mipNum = 1;
            nri::Descriptor* targetView = nullptr;
            TEST_CHECK(context.core.CreateTextureView(targetViewDesc, targetView));
            context.Track(targetView);
            nri::BufferDesc readbackDesc = {};
            readbackDesc.size = rowPitch * outputSize;
            nri::Buffer* readback = nullptr;
            TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

            nri::CommandAllocator* allocator = nullptr;
            nri::CommandBuffer* commands = nullptr;
            TEST_CHECK(context.CreateCommandObjects(queue, allocator, commands));
            TEST_CHECK(context.core.BeginCommandBuffer(*commands, pool));
            nri::TextureBarrierDesc barrier = {};
            barrier.texture = target;
            barrier.mipNum = 1;
            barrier.layerNum = 1;
            barrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
            nri::BarrierDesc barriers = {};
            barriers.textures = &barrier;
            barriers.textureNum = 1;
            context.core.CmdBarrier(*commands, barriers);
            nri::AttachmentDesc attachment = {};
            attachment.descriptor = targetView;
            attachment.loadOp = nri::LoadOp::CLEAR;
            attachment.storeOp = nri::StoreOp::STORE;
            nri::RenderingDesc rendering = {};
            rendering.colors = &attachment;
            rendering.colorNum = 1;
            context.core.CmdBeginRendering(*commands, rendering);
            context.core.CmdSetPipelineLayout(*commands, nri::BindPoint::GRAPHICS, *layout);
            context.core.CmdSetPipeline(*commands, *pipeline);
            context.core.CmdSetDescriptorSet(*commands, {0, set, nri::BindPoint::GRAPHICS});
            const nri::Viewport viewport = {0, 0, float(outputSize), float(outputSize), 0, 1};
            const nri::Rect scissor = {0, 0, outputSize, outputSize};
            context.core.CmdSetViewports(*commands, &viewport, 1);
            context.core.CmdSetScissors(*commands, &scissor, 1);
            context.core.CmdDraw(*commands, {3, 1, 0, 0});
            context.core.CmdEndRendering(*commands);
            barrier.before = barrier.after;
            barrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
            context.core.CmdBarrier(*commands, barriers);
            nri::TextureRegionDesc region = {};
            region.width = outputSize;
            region.height = outputSize;
            region.depth = 1;
            const nri::TextureDataLayoutDesc dataLayout = {0, rowPitch, rowPitch * outputSize};
            context.core.CmdReadbackTextureToBuffer(*commands, *readback, dataLayout, *target, region);
            TEST_CHECK(context.SubmitAndWait(queue, *commands));
            const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
            TEST_CHECK(data != nullptr);
            bool casePassed = true;
            for (uint32_t y = 0; y < outputSize; y++)
                for (uint32_t x = 0; x < outputSize; x++)
                    casePassed &= memcmp(data + y * rowPitch + x * 4, mipColors[biasIndex ? 0 : 1].data(), 4) == 0;
            if (!casePassed)
                printf("Sampler LOD bias output: %u,%u,%u,%u\n", data[0], data[1], data[2], data[3]);
            context.core.UnmapBuffer(*readback);
            const char* name = shaderKind ? (biasIndex ? "converted negative sampler LOD bias readback" : "converted positive sampler LOD bias readback") : (biasIndex ? "native negative sampler LOD bias readback" : "native positive sampler LOD bias readback");
            passed &= test::Report(name, casePassed);
        }
    }

    return passed;
}

bool Run(const test::Settings& settings) {
    if (settings.graphicsAPI != nri::GraphicsAPI::METAL) {
        printf("SKIP  MetalTests requires Metal\n");

        return true;
    }

    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    const nri::RootConstantDesc rootConstant = {0, sizeof(Constants), nri::StageBits::COMPUTE_SHADER};
    const nri::RootDescriptorDesc rootDescriptors[] = {
        {0, nri::DescriptorType::STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
        {0, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::COMPUTE_SHADER},
    };
    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.rootConstants = &rootConstant;
    layoutDesc.rootConstantNum = 1;
    layoutDesc.rootDescriptors = rootDescriptors;
    layoutDesc.rootDescriptorNum = 2;
    layoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    const uint64_t descriptorAlignment = std::max<uint64_t>(context.deviceDesc->memoryAlignment.bufferShaderResourceOffset, sizeof(uint32_t));
    const uint64_t descriptorOffset = descriptorAlignment;
    const uint32_t rootOffset = sizeof(uint32_t) * 2;
    std::vector<uint32_t> input(ELEMENT_NUM + GUARD_NUM);
    std::vector<uint32_t> initialOutput(ELEMENT_NUM + GUARD_NUM, SENTINEL);
    for (uint32_t i = 0; i < input.size(); i++)
        input[i] = i * 17 + 3;

    const Constants constants = {ELEMENT_NUM, 9, 0x12345, 0xA5A55A5A};
    nri::ShaderDesc nativeShader = LoadComputeShader(context, "MetalTests.metallib", "main0");
    nativeShader.threadGroupSizeX = THREAD_GROUP_SIZE;
    nativeShader.threadGroupSizeY = 1;
    nativeShader.threadGroupSizeZ = 1;
    TEST_CHECK(nativeShader.bytecode != nullptr);

    std::vector<uint32_t> nativeResult;
    TEST_CHECK(RunPipeline(context, *queue, *pipelineLayout, nativeShader, "native Metal compute", input, initialOutput, descriptorOffset, rootOffset, constants, nativeResult));

    if (context.deviceDesc->features.shaderBytecodeDXIL) {
        nri::ShaderDesc dxilShader = LoadComputeShader(context, "MetalTests.cs.dxil");
        TEST_CHECK(dxilShader.bytecode != nullptr);
        std::vector<uint32_t> dxilResult;
        TEST_CHECK(RunPipeline(context, *queue, *pipelineLayout, dxilShader, "HLSL/DXIL compute", input, initialOutput, descriptorOffset, rootOffset, constants, dxilResult));
        TEST_CHECK(test::Report("native Metal and DXIL equivalence", nativeResult == dxilResult));

        layoutDesc.rootDescriptorNum = 0;
        layoutDesc.flags = nri::PipelineLayoutBits::RESOURCE_HEAP_DIRECTLY_INDEXED;
        nri::PipelineLayout* heapLayout = nullptr;
        TEST_CHECK(context.core.CreatePipelineLayout(*context.device, layoutDesc, heapLayout));
        context.Track(heapLayout);
        nri::ShaderDesc heapShader = LoadComputeShader(context, "MetalTestsHeap.cs.dxil");
        TEST_CHECK(heapShader.bytecode != nullptr);
        std::vector<uint32_t> heapResult;
        TEST_CHECK(RunPipeline(context, *queue, *heapLayout, heapShader, "direct descriptor heap compute", input, initialOutput, descriptorOffset * 3, 0, constants, heapResult, true));
        TEST_CHECK(test::Report("descriptor heap and root descriptor equivalence", heapResult == nativeResult));

        const nri::DescriptorRangeDesc mutableRange = {0, 8, nri::DescriptorType::MUTABLE, nri::StageBits::COMPUTE_SHADER, nri::DescriptorRangeBits::PARTIALLY_BOUND};
        const nri::DescriptorSetDesc mutableSet = {0, &mutableRange, 1};
        nri::PipelineLayoutDesc mutableLayoutDesc = layoutDesc;
        mutableLayoutDesc.descriptorSets = &mutableSet;
        mutableLayoutDesc.descriptorSetNum = 1;
        nri::PipelineLayout* mutableLayout = nullptr;
        TEST_CHECK(context.core.CreatePipelineLayout(*context.device, mutableLayoutDesc, mutableLayout));
        context.Track(mutableLayout);
        std::vector<uint32_t> mutableResult;
        TEST_CHECK(RunPipeline(context, *queue, *mutableLayout, heapShader, "mutable descriptor update/copy/type-change compute", input, initialOutput, descriptorOffset * 3, 0, constants, mutableResult, false, true));
        TEST_CHECK(test::Report("mutable descriptors and descriptor heap equivalence", mutableResult == heapResult));
    } else {
        printf("SKIP  DXIL bytecode is unsupported\n");
    }

    bool clearPassed = TestBufferClears(context, *queue);

    nri::Color integerClear = {};
    integerClear.ui = {0x89ABCDEF, 0x10203040, 0x55667788, 0xDEADBEEF};
    clearPassed &= TestTextureClear(context, *queue, nri::Format::R32_UINT, integerClear, "CmdClearStorage integer texture subresource");
    clearPassed &= TestTextureClear(context, *queue, nri::Format::R32_UINT, integerClear, "CmdClearStorage 1D texture mip/layer", nri::TextureType::TEXTURE_1D);
    clearPassed &= TestTextureClear(context, *queue, nri::Format::R32_UINT, integerClear, "CmdClearStorage 3D remaining slices at nonzero mip/offset", nri::TextureType::TEXTURE_3D);

    nri::Color floatClear = {};
    floatClear.f = {-3.25f, 2.5f, 17.0f, -0.125f};
    clearPassed &= TestTextureClear(context, *queue, nri::Format::R32_SFLOAT, floatClear, "CmdClearStorage float texture");
    TEST_CHECK(clearPassed);
    for (uint32_t mode = 0; mode < 14; mode++)
        TEST_CHECK(TestResolve(context, *queue, mode));
    TEST_CHECK(TestNIS(context, *queue));
    TEST_CHECK(TestNativeLayerBasedMultiview(context, *queue));
    TEST_CHECK(TestAttachmentClears(context, *queue, nri::Format::R32_UINT));
    TEST_CHECK(TestAttachmentClears(context, *queue, nri::Format::R32_SINT));
    TEST_CHECK(TestAttachmentClears(context, *queue, nri::Format::R32_SINT, true));
    TEST_CHECK(TestAttachmentClears(context, *queue, nri::Format::D32_SFLOAT_S8_UINT));
    TEST_CHECK(TestNativeRayDispatch(context, *queue));
    TEST_CHECK(TestDynamicVertexStrides(context, *queue));
    TEST_CHECK(TestSamplerLodBias(context, *queue));
    TEST_CHECK(TestSamplerBorderColors(context));
    TEST_CHECK(TestWrapping(settings));

    nri::VideoMemoryInfo memoryInfo = {};
    TEST_CHECK(context.helper.QueryVideoMemoryInfo(*context.device, nri::MemoryLocation::DEVICE, memoryInfo));
    TEST_CHECK(test::Report("Metal memory budget and allocation usage", memoryInfo.budgetSize > 0 && memoryInfo.usageSize > 0));

    return true;
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
