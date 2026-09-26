// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include <array>

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::PipelineLayoutDesc graphicsLayoutDesc = {};
    graphicsLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    const bool testEmulation = context.deviceDesc->graphicsAPI == nri::GraphicsAPI::METAL;
    const nri::RootDescriptorDesc resultRoot = {0, nri::DescriptorType::STORAGE_STRUCTURED_BUFFER, nri::StageBits::VERTEX_SHADER};
    if (testEmulation) {
        graphicsLayoutDesc.flags = nri::PipelineLayoutBits::ENABLE_DRAW_PARAMETERS_EMULATION | nri::PipelineLayoutBits::ENABLE_DRAW_INDEX_EMULATION;
        graphicsLayoutDesc.rootDescriptors = &resultRoot;
        graphicsLayoutDesc.rootDescriptorNum = 1;
    }
    nri::PipelineLayout* graphicsLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, graphicsLayoutDesc, graphicsLayout));
    context.Track(graphicsLayout);

    nri::PipelineLayoutDesc computeLayoutDesc = {};
    computeLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::PipelineLayout* computeLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, computeLayoutDesc, computeLayout));
    context.Track(computeLayout);

    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    const nri::ShaderDesc graphicsShaders[] = {
        utils::LoadShader(graphicsAPI, graphicsAPI == nri::GraphicsAPI::METAL ? "IndirectCommandsEmulation.vs" : "IndirectCommands.vs", context.shaderStorage),
        utils::LoadShader(graphicsAPI, "IndirectCommands.fs", context.shaderStorage),
    };

    nri::ColorAttachmentDesc colorAttachmentDesc = {};
    colorAttachmentDesc.format = nri::Format::RGBA8_UNORM;
    colorAttachmentDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = graphicsLayout;
    graphicsPipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    graphicsPipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    graphicsPipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    graphicsPipelineDesc.outputMerger.colors = &colorAttachmentDesc;
    graphicsPipelineDesc.outputMerger.colorNum = 1;
    graphicsPipelineDesc.shaders = graphicsShaders;
    graphicsPipelineDesc.shaderNum = 2;

    nri::Pipeline* graphicsPipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, graphicsPipeline));
    context.Track(graphicsPipeline);

    nri::ComputePipelineDesc computePipelineDesc = {};
    computePipelineDesc.pipelineLayout = computeLayout;
    computePipelineDesc.shader = utils::LoadShader(graphicsAPI, "IndirectCommands.cs", context.shaderStorage);
    nri::Pipeline* computePipeline = nullptr;
    TEST_CHECK(context.core.CreateComputePipeline(*context.device, computePipelineDesc, computePipeline));
    context.Track(computePipeline);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.format = colorAttachmentDesc.format;
    textureDesc.width = 32;
    textureDesc.height = 32;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureViewDesc textureViewDesc = {};
    textureViewDesc.texture = texture;
    textureViewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
    textureViewDesc.format = textureDesc.format;
    textureViewDesc.mipNum = 1;
    textureViewDesc.layerNum = 1;
    textureViewDesc.sliceNum = 1;
    nri::Descriptor* colorAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(textureViewDesc, colorAttachment));
    context.Track(colorAttachment);

    struct Arguments {
        nri::DrawBaseDesc draw;
        uint8_t padding0[32 - sizeof(nri::DrawBaseDesc)];
        nri::DrawBaseDesc draw2;
        uint8_t padding1[32 - sizeof(nri::DrawBaseDesc)];
        nri::DrawIndexedBaseDesc drawIndexed;
        uint8_t padding2[32 - sizeof(nri::DrawIndexedBaseDesc)];
        nri::DrawIndexedBaseDesc drawIndexed2;
        uint8_t padding3[32 - sizeof(nri::DrawIndexedBaseDesc)];
        nri::DrawIndexedBaseDesc counted;
        uint8_t padding4[32 - sizeof(nri::DrawIndexedBaseDesc)];
        nri::DrawIndexedBaseDesc suppressed;
        uint8_t padding5[32 - sizeof(nri::DrawIndexedBaseDesc)];
        nri::DispatchDesc dispatch;
    } arguments = {};

    arguments.draw = {5, 11, 3, 1, 5, 11};
    arguments.draw2 = {8, 13, 3, 1, 8, 13};
    arguments.drawIndexed = {4, 17, 3, 1, 0, 4, 17};
    arguments.drawIndexed2 = {-2, 22, 3, 1, 0, -2, 22};
    arguments.counted = {31, 26, 3, 1, 0, 31, 26};
    arguments.suppressed = {2, 29, 3, 1, 0, 2, 29};
    arguments.dispatch = {1, 1, 1};

    nri::BufferDesc argumentBufferDesc = {};
    argumentBufferDesc.size = sizeof(arguments);
    argumentBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
    nri::Buffer* argumentBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentBufferDesc, nri::MemoryLocation::DEVICE, argumentBuffer));

    const uint16_t indices[] = {0, 1, 2, 0}; // WGPU: source offset, destination offset, and copy size must be multiples of 4
    nri::BufferDesc indexBufferDesc = {};
    indexBufferDesc.size = sizeof(indices);
    indexBufferDesc.usage = nri::BufferUsageBits::INDEX;
    nri::Buffer* indexBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(indexBufferDesc, nri::MemoryLocation::DEVICE, indexBuffer));

    const uint32_t indirectCount[] = {0, 1};
    nri::Buffer* countBuffer = nullptr;
    if (context.deviceDesc->features.drawIndirectCount) {
        nri::BufferDesc countBufferDesc = {};
        countBufferDesc.size = sizeof(indirectCount);
        countBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
        TEST_CHECK(context.CreateBuffer(countBufferDesc, nri::MemoryLocation::DEVICE, countBuffer));
    }

    const nri::BufferUploadDesc uploads[] = {
        {&arguments, argumentBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
        {indices, indexBuffer, {nri::AccessBits::INDEX_BUFFER, nri::StageBits::INDEX_INPUT}},
    };
    TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, uploads, 2));
    if (countBuffer) {
        const nri::BufferUploadDesc countUpload = {&indirectCount, countBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}};
        TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, &countUpload, 1));
    }

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    std::array<uint32_t, 32 * 3 * 4> expected = {};
    nri::Buffer* results = nullptr;
    nri::Buffer* readback = nullptr;
    nri::Descriptor* resultView = nullptr;
    if (testEmulation) {
        nri::BufferDesc resultDesc = {};
        resultDesc.size = sizeof(expected);
        resultDesc.structureStride = 16;
        resultDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;
        TEST_CHECK(context.CreateBuffer(resultDesc, nri::MemoryLocation::DEVICE, results));
        resultDesc.usage = nri::BufferUsageBits::NONE;
        TEST_CHECK(context.CreateBuffer(resultDesc, nri::MemoryLocation::HOST_READBACK, readback));
        nri::BufferViewDesc view = {};
        view.buffer = results;
        view.type = nri::BufferView::STORAGE_STRUCTURED_BUFFER;
        view.size = sizeof(expected);
        view.structureStride = 16;
        TEST_CHECK(context.core.CreateBufferView(view, resultView));
        context.Track(resultView);
        const nri::BufferUploadDesc upload = {expected.data(), results, {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::VERTEX_SHADER}};
        TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, &upload, 1));
    }
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
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

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *graphicsLayout);
    context.core.CmdSetPipeline(*commandBuffer, *graphicsPipeline);
    if (testEmulation)
        context.core.CmdSetRootDescriptor(*commandBuffer, {0, resultView, 0, nri::BindPoint::GRAPHICS});
    const nri::Viewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, 32, 32};
    context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
    context.core.CmdSetIndexBuffer(*commandBuffer, *indexBuffer, 0, nri::IndexType::UINT16);
    context.core.CmdDraw(*commandBuffer, {3, 1, 2, 3});
    context.core.CmdDrawIndexed(*commandBuffer, {3, 1, 0, 6, 7});
    const uint32_t drawStride = 32;
    const uint32_t drawIndexedStride = 32;
    const uint64_t drawOffset = graphicsAPI == nri::GraphicsAPI::METAL ? 0 : 8;
    const uint64_t drawIndexedOffset = graphicsAPI == nri::GraphicsAPI::METAL ? 64 : 72;
    context.core.CmdDrawIndirect(*commandBuffer, *argumentBuffer, drawOffset, 2, drawStride, nullptr, 0);
    context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, drawIndexedOffset, 2, drawIndexedStride, nullptr, 0);
    if (context.deviceDesc->features.drawIndirectCount) {
        context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, drawIndexedOffset + 64, 2, drawIndexedStride, countBuffer, sizeof(uint32_t));
        context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, drawIndexedOffset + 96, 1, drawIndexedStride, countBuffer, 0);
    }
    context.core.CmdEndRendering(*commandBuffer);

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *computeLayout);
    context.core.CmdSetPipeline(*commandBuffer, *computePipeline);
    context.core.CmdDispatchIndirect(*commandBuffer, *argumentBuffer, offsetof(Arguments, dispatch));

    if (testEmulation) {
        nri::BufferBarrierDesc barrier = {};
        barrier.buffer = results;
        barrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::VERTEX_SHADER};
        barrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
        nri::BarrierDesc barriers = {};
        barriers.buffers = &barrier;
        barriers.bufferNum = 1;
        context.core.CmdBarrier(*commandBuffer, barriers);
        context.core.CmdCopyBuffer(*commandBuffer, *readback, 0, *results, 0, sizeof(expected));
    }
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    if (testEmulation) {
        const int32_t cases[][3] = {{2, 3, 0}, {6, 7, 0}, {5, 11, 0}, {8, 13, 1}, {4, 17, 0}, {-2, 22, 1}, {31, 26, 0}};
        for (const auto& c : cases) {
            for (uint32_t vertex = 0; vertex < 3; vertex++) {
                const uint32_t offset = ((c[1] + c[2]) * 3 + vertex) * 4;
                expected[offset] = c[0];
                expected[offset + 1] = c[1];
                expected[offset + 2] = c[2];
                expected[offset + 3] = uint32_t(c[0]) + vertex;
            }
        }
        const void* actual = context.core.MapBuffer(*readback, 0, sizeof(expected));
        TEST_CHECK(actual != nullptr);
        const bool matches = memcmp(actual, expected.data(), sizeof(expected)) == 0;
        context.core.UnmapBuffer(*readback);
        TEST_CHECK(test::Report("draw bases and indices, direct/indexed/indirect readback", matches));
    }
    return test::Report("indirect draw and dispatch", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
