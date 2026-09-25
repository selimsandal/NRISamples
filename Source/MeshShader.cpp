// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include "Extensions/NRIMeshShader.h"

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    if (!context.deviceDesc->features.meshShader) {
        printf("SKIP  Mesh shaders are unsupported\n");

        return true;
    }

    nri::MeshShaderInterface mesh = {};
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::MeshShaderInterface), &mesh));

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.shaderStages = nri::StageBits::MESH_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    const nri::ShaderDesc shaders[] = {
        utils::LoadShader(context.deviceDesc->graphicsAPI, "MeshShader.ms", context.shaderStorage),
        utils::LoadShader(context.deviceDesc->graphicsAPI, "MeshShader.fs", context.shaderStorage),
    };

    nri::ColorAttachmentDesc colorDesc = {};
    colorDesc.format = nri::Format::RGBA8_UNORM;
    colorDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = pipelineLayout;
    graphicsPipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    graphicsPipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    graphicsPipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    graphicsPipelineDesc.outputMerger.colors = &colorDesc;
    graphicsPipelineDesc.outputMerger.colorNum = 1;
    graphicsPipelineDesc.shaders = shaders;
    graphicsPipelineDesc.shaderNum = 2;

    struct Cache {
        const nri::CoreInterface& core;
        nri::PipelineCache* object = nullptr;

        ~Cache() {
            core.DestroyPipelineCache(object);
        }
    } cache{context.core};

    const nri::PipelineCacheDesc emptyCache = {};
    TEST_CHECK(context.core.CreatePipelineCache(*context.device, emptyCache, cache.object));
    graphicsPipelineDesc.cache = cache.object;
    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, pipeline));
    context.Track(pipeline);

    uint64_t cacheSize = 0;
    TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, nullptr, cacheSize));
    std::vector<uint8_t> cacheData(cacheSize);
    TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, cacheData.data(), cacheSize));
    context.core.DestroyPipelineCache(cache.object);
    cache.object = nullptr;
    const nri::PipelineCacheDesc savedCache = {cacheData.data(), cacheSize};
    TEST_CHECK(context.core.CreatePipelineCache(*context.device, savedCache, cache.object));
    graphicsPipelineDesc.cache = cache.object;
    if (context.deviceDesc->features.pipelineCacheControl)
        graphicsPipelineDesc.flags = nri::GraphicsPipelineBits::FAIL_ON_CACHE_MISS;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, pipeline));
    context.Track(pipeline);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.format = colorDesc.format;
    textureDesc.width = 32;
    textureDesc.height = 32;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = texture;
    viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
    viewDesc.format = textureDesc.format;
    viewDesc.mipNum = 1;
    viewDesc.layerNum = 1;
    viewDesc.sliceNum = 1;
    nri::Descriptor* colorAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, colorAttachment));
    context.Track(colorAttachment);

    const nri::DrawMeshTasksDesc arguments = {1, 1, 1};
    const uint32_t argumentData[] = {0, 0, 0, 0, 1, 1, 1, 0xBAD, 1, 1, 1, 0xBAD};
    const uint32_t countData[] = {99, 0, 1, 5};
    nri::BufferDesc argumentBufferDesc = {};
    argumentBufferDesc.size = sizeof(argumentData);
    argumentBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
    nri::Buffer* argumentBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentBufferDesc, nri::MemoryLocation::DEVICE, argumentBuffer));

    argumentBufferDesc.size = sizeof(countData);
    nri::Buffer* countBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentBufferDesc, nri::MemoryLocation::DEVICE, countBuffer));
    const nri::BufferUploadDesc uploadDescs[] = {
        {argumentData, argumentBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
        {countData, countBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
    };
    TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, uploadDescs, 2));

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));

    nri::TextureDataLayoutDesc readbackLayout = {};
    readbackLayout.rowPitch = 256;
    readbackLayout.slicePitch = readbackLayout.rowPitch * 32;
    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = readbackLayout.slicePitch;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::QueryPool* occlusionPools[2] = {};
    nri::Buffer* queryReadback = nullptr;
    if (context.deviceDesc->features.occlusion) {
        const nri::QueryPoolDesc poolDesc = {nri::QueryType::OCCLUSION, 1};
        for (auto& pool : occlusionPools) {
            TEST_CHECK(context.core.CreateQueryPool(*context.device, poolDesc, pool));
            context.Track(pool);
        }
        nri::BufferDesc queryBufferDesc = {};
        queryBufferDesc.size = 2 * sizeof(uint64_t);
        TEST_CHECK(context.CreateBuffer(queryBufferDesc, nri::MemoryLocation::HOST_READBACK, queryReadback));
    }

    const uint32_t expectedDraws[] = {1, 1, 2, 0, 1, 2};
    const char* cases[] = {"direct", "indirect", "multi-draw", "count zero", "count one", "count clamped"};
    for (uint32_t indirect = 0; indirect < (context.deviceDesc->features.drawIndirectCount ? 6u : 3u); indirect++) {
        printf("Mesh case: %s\n", cases[indirect]);
        if (indirect)
            context.core.ResetCommandAllocator(*commandAllocator);
        TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

        if (queryReadback) {
            for (auto pool : occlusionPools)
                context.core.CmdResetQueries(*commandBuffer, *pool, 0, 1);
        }

        nri::TextureBarrierDesc textureBarrier = {};
        textureBarrier.texture = texture;
        if (indirect)
            textureBarrier.before = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
        textureBarrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
        textureBarrier.mipNum = 1;
        textureBarrier.layerNum = 1;
        nri::BarrierDesc barrierDesc = {};
        barrierDesc.textures = &textureBarrier;
        barrierDesc.textureNum = 1;
        context.core.CmdBarrier(*commandBuffer, barrierDesc);

        nri::AttachmentDesc attachmentDesc = {};
        attachmentDesc.descriptor = colorAttachment;
        attachmentDesc.loadOp = nri::LoadOp::CLEAR;
        attachmentDesc.storeOp = nri::StoreOp::STORE;
        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colors = &attachmentDesc;
        renderingDesc.colorNum = 1;
        context.core.CmdBeginRendering(*commandBuffer, renderingDesc);
        context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *pipelineLayout);
        context.core.CmdSetPipeline(*commandBuffer, *pipeline);
        const nri::Viewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
        const nri::Rect scissor = {0, 0, 32, 32};
        context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
        context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
        if (queryReadback)
            context.core.CmdBeginQuery(*commandBuffer, *occlusionPools[0], 0);
        if (indirect)
            mesh.CmdDrawMeshTasksIndirect(*commandBuffer, *argumentBuffer, 16, indirect == 1 ? 1 : 2, 16, indirect >= 3 ? countBuffer : nullptr, indirect >= 3 ? (indirect - 2) * sizeof(uint32_t) : 0);
        else
            mesh.CmdDrawMeshTasks(*commandBuffer, arguments);
        if (queryReadback) {
            context.core.CmdEndQuery(*commandBuffer, *occlusionPools[0], 0);
            // Switching pools must preserve the triangle already rendered.
            context.core.CmdBeginQuery(*commandBuffer, *occlusionPools[1], 0);
            context.core.CmdEndQuery(*commandBuffer, *occlusionPools[1], 0);
        }
        context.core.CmdEndRendering(*commandBuffer);
        if (queryReadback) {
            for (uint32_t i = 0; i < 2; i++)
                context.core.CmdCopyQueries(*commandBuffer, *occlusionPools[i], 0, 1, *queryReadback, i * sizeof(uint64_t));
        }
        textureBarrier.before = textureBarrier.after;
        textureBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
        context.core.CmdBarrier(*commandBuffer, barrierDesc);
        nri::TextureRegionDesc region = {};
        region.width = 32;
        region.height = 32;
        region.depth = 1;
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, readbackLayout, *texture, region);
        TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

        const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, readbackDesc.size);
        TEST_CHECK(data != nullptr);
        // Pixel (16,16) has barycentric weights (7/32,25/96,25/48).
        // Allow one UNORM step for interpolation and conversion rounding.
        const uint8_t* center = data + 16 * readbackLayout.rowPitch + 16 * 4;
        const uint8_t expected[] = {56, 66, 133, 255};
        bool passed = true;
        for (uint32_t channel = 0; channel < 4; channel++) {
            passed &= abs(int(center[channel]) - (expectedDraws[indirect] ? int(expected[channel]) : 0)) <= 1;
            passed &= data[channel] == 0;
        }
        context.core.UnmapBuffer(*readback);
        TEST_CHECK(test::Report(indirect ? "mesh indirect draw pixel readback" : "mesh direct draw pixel readback", passed));
        if (queryReadback) {
            const uint64_t* counts = (const uint64_t*)context.core.MapBuffer(*queryReadback, 0, nri::WHOLE_SIZE);
            // The single-sample triangle covers half of a 24-by-24 pixel square.
            passed = counts && counts[0] == 288 * expectedDraws[indirect] && counts[1] == 0;
            context.core.UnmapBuffer(*queryReadback);
            TEST_CHECK(test::Report(indirect ? "mesh indirect occlusion counts" : "mesh direct occlusion counts", passed));
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
