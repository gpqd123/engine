#pragma once

// ============================================================
//  RendererSystem.hpp
//  把 main.cpp 的所有逻辑原封不动搬进来
//  Init()     = main() 里主循环之前的全部代码
//  Update(dt) = 主循环的单次迭代
//  Shutdown() = vkDeviceWaitIdle
// ============================================================

#include "../Core/System.hpp"

#include "../../../ThirdParty/volk/include/volk/volk.h"
//#include <volk/volk.h>

#include <print>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE

#include "../../../ThirdParty/glfw/include/GLFW/glfw3.h"
//#include <GLFW/glfw3.h>




#if !defined(GLM_FORCE_RADIANS)
#   define GLM_FORCE_RADIANS
#endif

#include "../../../ThirdParty/glm/include/glm/glm.hpp"
//#include <glm/glm.hpp>

#include "../../../ThirdParty/glm/include/glm/gtx/transform.hpp"
//#include <glm/gtx/transform.hpp>

#include "../../../ThirdParty/glm/include/glm/gtc/matrix_transform.hpp"
//#include <glm/gtc/matrix_transform.hpp>

#include "../Rhi/angle.hpp"
using namespace labut2::literals;

#include "../Rhi/load.hpp"
#include "../Rhi/error.hpp"
#include "../Rhi/synch.hpp"
#include "../Rhi/vkimage.hpp"
#include "../Rhi/commands.hpp"
#include "../Rhi/textures.hpp"
#include "../Rhi/vkbuffer.hpp"
#include "../Rhi/vkobject.hpp"
#include "../Rhi/to_string.hpp"
#include "../Rhi/descriptors.hpp"
#include "../Rhi/vulkan_window.hpp"
namespace lut = labut2;

#include "RenderUtilities/engine_model.hpp"
#include "RenderUtilities/camera.hpp"
#include "RenderUtilities/setup.hpp"
#include "RenderUtilities/rendering.hpp"

namespace glsl {
    struct MosaicUniform {
        int   mosaicOn;
        float pad[3];
    };
}

namespace engine {

    class RendererSystem final : public System
    {
    public:
        explicit RendererSystem(bool& appRunning)
            : mAppRunning(appRunning) {
        }

        // ─────────────────────────────────────────────────────────
        //  Init
        // ─────────────────────────────────────────────────────────
        void Init() override
        {
            // 1. 窗口
            mWindow = lut::make_vulkan_window();

            // 2. GLFW 回调
            glfwSetWindowUserPointer(mWindow.window, &mState);
            glfwSetKeyCallback(mWindow.window, &glfw_callback_key_press);
            glfwSetMouseButtonCallback(mWindow.window, &glfw_callback_button);
            glfwSetCursorPosCallback(mWindow.window, &glfw_callback_motion);
       
            // 3. VMA allocator
            mAllocator = lut::create_allocator(mWindow);

            // 4. Descriptor layouts
            mSceneLayout = create_scene_descriptor_layout(mWindow);
            mObjectLayout = create_object_descriptor_layout(mWindow);
            mPostLayout = create_post_proc_descriptor_layout(mWindow);

            // 5. Pipeline layouts
            mPipeLayout = create_triangle_pipeline_layout(mWindow, mSceneLayout.handle, mObjectLayout.handle);
            mPostPipeLayout = create_post_proc_pipeline_layout(mWindow, mPostLayout.handle);

            // 6. Pipelines（初始一批）
            mPipe = create_triangle_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R16G16B16A16_SFLOAT);
            mMipPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugMipFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
            mDepthPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugDepthFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
            mDerivPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugDerivFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
            mOverdrawPipe = create_overdraw_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R8G8B8A8_UNORM);
            mOvershadingPipe = create_overshading_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R8G8B8A8_UNORM);
            mVisResolvePipe = create_vis_resolve_pipeline(mWindow, mPostPipeLayout.handle, mPostLayout.handle);

            // 7. Command pool + per-frame 同步对象
            mCmdPool = lut::create_command_pool(mWindow,
                VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

            for (std::size_t i = 0; i < mWindow.swapImages.size(); ++i) {
                mCmdBuffers.emplace_back(lut::alloc_command_buffer(mWindow, mCmdPool.handle));
                mFrameDone.emplace_back(lut::create_fence(mWindow.device, VK_FENCE_CREATE_SIGNALED_BIT));
                mImageAvailable.emplace_back(lut::create_semaphore(mWindow.device));
                mRenderFinished.emplace_back(lut::create_semaphore(mWindow.device));
            }

            // 8. 加载模型（CPU 端，EngineModel）
            mModel = load_engine_model_glb("../../../Assets/Models/TScene.glb");

            // 9. 上传纹理（使用 lut::load_image_texture2d_from_memory）
            for (auto const& tex : mModel.textures) {
                glfwPollEvents();

                VkFormat fmt = (tex.space == ETextureSpace::srgb)
                    ? VK_FORMAT_R8G8B8A8_SRGB
                    : VK_FORMAT_R8G8B8A8_UNORM;

                mModelTextures.emplace_back(
                    lut::load_image_texture2d_from_memory(
                        tex.pixels.data(),
                        static_cast<uint32_t>(tex.width),
                        static_cast<uint32_t>(tex.height),
                        mWindow, mCmdPool.handle, mAllocator, fmt));

                mModelTextureViews.emplace_back(
                    lut::create_image_view_texture2d(mWindow, mModelTextures.back().image, fmt));
            }

            // 10. 默认灰色纹理（材质缺纹理时占位）
            {
                std::uint8_t grey[4] = { 128, 128, 128, 255 };
                mDefaultGrayTex = lut::load_image_texture2d_from_memory(
                    grey, 1, 1,
                    mWindow, mCmdPool.handle, mAllocator,
                    VK_FORMAT_R8G8B8A8_UNORM);
                mDefaultGrayView = lut::create_image_view_texture2d(
                    mWindow, mDefaultGrayTex.image, VK_FORMAT_R8G8B8A8_UNORM);
            }

            // 11. Samplers + Descriptor pool
            mDefaultSampler = lut::create_default_sampler(mWindow);
            mDebugSampler = create_debug_sampler(mWindow);
            mDescPool = lut::create_descriptor_pool(mWindow);

            // 12. 材质描述符（普通 + debug）
            BuildMaterialDescriptors(mDefaultSampler.handle, mMaterialDescriptors);
            BuildMaterialDescriptors(mDebugSampler.handle, mDebugMaterialDescriptors);

            // 13. 上传 Mesh
            UploadMeshes();

            // 14. Scene UBO + scene descriptor（binding 0）
            mSceneUBO = lut::create_buffer(mAllocator,
                sizeof(glsl::SceneUniform),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                0,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

            mSceneDescriptors = lut::alloc_desc_set(mWindow, mDescPool.handle, mSceneLayout.handle);
            {
                VkDescriptorBufferInfo bi{ mSceneUBO.buffer, 0, VK_WHOLE_SIZE };
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = mSceneDescriptors; w.dstBinding = 0;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w.descriptorCount = 1; w.pBufferInfo = &bi;
                vkUpdateDescriptorSets(mWindow.device, 1, &w, 0, nullptr);
            }

            // 15. Alpha pipeline
            mAlphaPipe = create_alpha_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R16G16B16A16_SFLOAT);

            // 16. Shadow resources
            mShadowMap = create_shadow_map(mWindow, mAllocator);
            mShadowSampler = create_shadow_sampler(mWindow);
            mShadowPipe = create_shadow_pipeline(mWindow, mPipeLayout.handle);

            // 17. Depth / offscreen / vis
            mDepthBuffer = create_depth_buffer(mWindow, mAllocator);
            mPostProcPipe = create_post_proc_pipeline(mWindow, mPostPipeLayout.handle, mPostLayout.handle);
            mOffscreenImage = create_offscreen_buffer(mWindow, mAllocator);
            mVisImage = create_vis_image(mWindow, mAllocator);
            mPostSampler = create_post_proc_sampler(mWindow);

            // 18. Scene descriptor 更新：加入 shadow map（binding 1）
            {
                VkDescriptorBufferInfo bi{ mSceneUBO.buffer, 0, VK_WHOLE_SIZE };
                VkDescriptorImageInfo  si{};
                si.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
                si.imageView = mShadowMap.view;
                si.sampler = mShadowSampler.handle;

                VkWriteDescriptorSet w[2]{};
                w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[0].dstSet = mSceneDescriptors; w[0].dstBinding = 0;
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w[0].descriptorCount = 1; w[0].pBufferInfo = &bi;

                w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[1].dstSet = mSceneDescriptors; w[1].dstBinding = 1;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[1].descriptorCount = 1; w[1].pImageInfo = &si;

                vkUpdateDescriptorSets(mWindow.device, 2, w, 0, nullptr);
            }

            // 19. Per-frame mosaic UBO + post-proc / vis descriptor sets
            for (std::size_t i = 0; i < mCmdBuffers.size(); ++i) {
                mMosaicUBOs.emplace_back(lut::create_buffer(mAllocator,
                    sizeof(glsl::MosaicUniform),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

                mPostDescriptors.emplace_back(
                    BuildPostDesc(mOffscreenImage.view, mMosaicUBOs.back().buffer));

                mVisDescriptors.emplace_back(
                    BuildPostDesc(mVisImage.view, mMosaicUBOs[i].buffer));
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Update
        // ─────────────────────────────────────────────────────────
        void Update(float dt) override
        {
            glfwPollEvents();

            if (glfwWindowShouldClose(mWindow.window)) {
                mAppRunning = false;
                return;
            }

            // ── Swapchain 重建 ──
            if (mRecreateSwapchain) {
                vkDeviceWaitIdle(mWindow.device);
                auto const changes = lut::recreate_swapchain(mWindow);

                if (changes.changedFormat) {
                    mPipe = create_triangle_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R16G16B16A16_SFLOAT);
                    mAlphaPipe = create_alpha_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R16G16B16A16_SFLOAT);
                    mMipPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugMipFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
                    mDepthPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugDepthFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
                    mDerivPipe = create_debug_pipeline(mWindow, mPipeLayout.handle, cfg::kDebugVertShaderPath, cfg::kDebugDerivFragShaderPath, VK_FORMAT_R16G16B16A16_SFLOAT);
                    mPostProcPipe = create_post_proc_pipeline(mWindow, mPostPipeLayout.handle, mPostLayout.handle);
                    mOverdrawPipe = create_overdraw_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R8G8B8A8_UNORM);
                    mOvershadingPipe = create_overshading_pipeline(mWindow, mPipeLayout.handle, VK_FORMAT_R8G8B8A8_UNORM);
                    mVisResolvePipe = create_vis_resolve_pipeline(mWindow, mPostPipeLayout.handle, mPostLayout.handle);
                }

                if (changes.changedSize) {
                    mDepthBuffer = create_depth_buffer(mWindow, mAllocator);
                    mOffscreenImage = create_offscreen_buffer(mWindow, mAllocator);
                    mVisImage = create_vis_image(mWindow, mAllocator);

                    UpdatePostDescImage(mPostDescriptors, mOffscreenImage.view);
                    UpdatePostDescImage(mVisDescriptors, mVisImage.view);
                }

                mRecreateSwapchain = false;
                return;
            }

            // ── 正常渲染帧 ──
            mFrameIndex = (mFrameIndex + 1) % mCmdBuffers.size();

            if (auto res = vkWaitForFences(mWindow.device, 1,
                &mFrameDone[mFrameIndex].handle, VK_TRUE,
                std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
                throw lut::Error("vkWaitForFences: {}", lut::to_string(res));

            std::uint32_t imageIndex = 0;
            auto acquireRes = vkAcquireNextImageKHR(
                mWindow.device, mWindow.swapchain,
                std::numeric_limits<std::uint64_t>::max(),
                mImageAvailable[mFrameIndex].handle,
                VK_NULL_HANDLE, &imageIndex);

            if (acquireRes == VK_SUBOPTIMAL_KHR || acquireRes == VK_ERROR_OUT_OF_DATE_KHR) {
                mRecreateSwapchain = true;
                mFrameIndex = (mFrameIndex + mCmdBuffers.size() - 1) % mCmdBuffers.size();
                return;
            }
            if (acquireRes != VK_SUCCESS)
                throw lut::Error("vkAcquireNextImageKHR: {}", lut::to_string(acquireRes));

            if (auto res = vkResetFences(mWindow.device, 1,
                &mFrameDone[mFrameIndex].handle); VK_SUCCESS != res)
                throw lut::Error("vkResetFences: {}", lut::to_string(res));

            // 更新相机状态
            update_user_state(mState, dt);

            // 组装 SceneUniform
            glsl::SceneUniform sceneUniforms{};
            update_scene_uniforms(sceneUniforms,
                mWindow.swapchainExtent.width,
                mWindow.swapchainExtent.height,
                mState);

            // 选管线
            VkPipeline  currentOpaque = mPipe.handle;
            VkPipeline  currentAlpha = mAlphaPipe.handle;
            auto const* currentDescs = &mMaterialDescriptors;

            switch (mState.renderMode) {
            case 1: currentOpaque = currentAlpha = mMipPipe.handle;
                currentDescs = &mDebugMaterialDescriptors; break;
            case 2: currentOpaque = currentAlpha = mDepthPipe.handle;
                currentDescs = &mDebugMaterialDescriptors; break;
            case 3: currentOpaque = currentAlpha = mDerivPipe.handle;
                currentDescs = &mDebugMaterialDescriptors; break;
            case 4: currentOpaque = currentAlpha = mOverdrawPipe.handle;
                currentDescs = &mDebugMaterialDescriptors; break;
            case 5: currentOpaque = currentAlpha = mOvershadingPipe.handle;
                currentDescs = &mDebugMaterialDescriptors; break;
            default: break;
            }

            // 选 offscreen target
            ImageAndView    offscreenTarget;
            VkPipeline      resolvePipeline = mPostProcPipe.handle;
            VkDescriptorSet resolveDescs = mPostDescriptors[mFrameIndex];
            VkPipelineLayout resolveLayout = mPostPipeLayout.handle;
            VkClearColorValue clearColor = { 0.1f, 0.1f, 0.1f, 1.f };

            if (mState.renderMode == 4 || mState.renderMode == 5) {
                offscreenTarget = { mVisImage.image, mVisImage.view };
                resolvePipeline = mVisResolvePipe.handle;
                resolveDescs = mVisDescriptors[mFrameIndex];
                clearColor = { 0.f, 0.1f, 0.f, 1.f };
            }
            else {
                offscreenTarget = { mOffscreenImage.image, mOffscreenImage.view };
            }

            // Mosaic UBO
            {
                glsl::MosaicUniform mu{ mState.mosaicEnabled ? 1 : 0, {} };
                void* ptr;
                vmaMapMemory(mAllocator.allocator, mMosaicUBOs[mFrameIndex].allocation, &ptr);
                std::memcpy(ptr, &mu, sizeof(mu));
                vmaUnmapMemory(mAllocator.allocator, mMosaicUBOs[mFrameIndex].allocation);
            }

            // Record
            ImageAndView colorTarget = { mWindow.swapImages[imageIndex], mWindow.swapViews[imageIndex] };
            ImageAndView depthTarget = { mDepthBuffer.image, mDepthBuffer.view };
            ImageAndView shadowTarget = { mShadowMap.image,   mShadowMap.view };

            record_commands(
                mCmdBuffers[mFrameIndex],
                currentOpaque, currentAlpha,
                colorTarget, depthTarget,
                mWindow.swapchainExtent,
                mSceneUBO.buffer, sceneUniforms,
                mPipeLayout.handle, mSceneDescriptors,
                mMeshPositions, mMeshTexCoords, mMeshNormals, mMeshIndices,
                mModel.meshes, mModel.materials,
                *currentDescs,
                mModel.scenes,           // ← 新增：原 main.cpp 里多传的参数
                resolvePipeline, resolveDescs, resolveLayout,
                offscreenTarget, clearColor,
                mShadowPipe.handle, shadowTarget
            );

            submit_commands(mWindow,
                mCmdBuffers[mFrameIndex],
                mFrameDone[mFrameIndex].handle,
                mImageAvailable[mFrameIndex].handle,
                mRenderFinished[mFrameIndex].handle);

            present_results(mWindow.presentQueue, mWindow.swapchain,
                imageIndex, mRenderFinished[mFrameIndex].handle,
                mRecreateSwapchain);
        }

        void Shutdown() override
        {
            vkDeviceWaitIdle(mWindow.device);
        }

    private:
        // ─────────────────────────────────────────────────────────
        //  材质描述符构建（baseColorTexture / metalRoughTexture 带 -1 检查）
        // ─────────────────────────────────────────────────────────
        void BuildMaterialDescriptors(VkSampler sampler,
            std::vector<VkDescriptorSet>& out)
        {
            for (auto const& mat : mModel.materials) {
                VkDescriptorSet ds = lut::alloc_desc_set(
                    mWindow, mDescPool.handle, mObjectLayout.handle);

                // 如果材质没有对应纹理，使用默认灰色
                VkImageView baseView = mDefaultGrayView.handle;
                if (mat.baseColorTexture >= 0)
                    baseView = mModelTextureViews[mat.baseColorTexture].handle;

                VkImageView mrView = mDefaultGrayView.handle;
                if (mat.metalRoughTexture >= 0)
                    mrView = mModelTextureViews[mat.metalRoughTexture].handle;

                VkDescriptorImageInfo imgs[3]{};
                imgs[0] = { sampler, baseView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                imgs[1] = { sampler, mrView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                imgs[2] = { sampler, mrView,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

                VkWriteDescriptorSet w[3]{};
                for (int j = 0; j < 3; ++j) {
                    w[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[j].dstSet = ds; w[j].dstBinding = (uint32_t)j;
                    w[j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w[j].descriptorCount = 1; w[j].pImageInfo = &imgs[j];
                }
                vkUpdateDescriptorSets(mWindow.device, 3, w, 0, nullptr);
                out.emplace_back(ds);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Mesh 上传（单次 command buffer）
        // ─────────────────────────────────────────────────────────
        void UploadMeshes()
        {
            VkCommandBuffer uploadCmd = lut::alloc_command_buffer(mWindow, mCmdPool.handle);
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(uploadCmd, &bi);

            std::vector<lut::Buffer> staging;

            for (auto const& mesh : mModel.meshes) {
                glfwPollEvents();

                VkDeviceSize posSz = mesh.positions.size() * sizeof(glm::vec3);
                VkDeviceSize texSz = mesh.texcoords.size() * sizeof(glm::vec2);
                VkDeviceSize normSz = mesh.normals.size() * sizeof(glm::vec3);
                VkDeviceSize idxSz = mesh.indices.size() * sizeof(std::uint32_t);

                // GPU buffers
                auto mkGpu = [&](VkDeviceSize sz, VkBufferUsageFlags usage) {
                    return lut::create_buffer(mAllocator, sz,
                        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
                    };
                mMeshPositions.emplace_back(mkGpu(posSz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
                mMeshTexCoords.emplace_back(mkGpu(texSz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
                mMeshNormals.emplace_back(mkGpu(normSz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
                mMeshIndices.emplace_back(mkGpu(idxSz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT));

                // Staging buffers
                auto mkStg = [&](VkDeviceSize sz) {
                    return lut::create_buffer(mAllocator, sz,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    };
                lut::Buffer ps = mkStg(posSz), ts = mkStg(texSz),
                    ns = mkStg(normSz), is = mkStg(idxSz);

                // memcpy
                auto up = [&](lut::Buffer& b, const void* src, VkDeviceSize sz) {
                    void* ptr;
                    vmaMapMemory(mAllocator.allocator, b.allocation, &ptr);
                    std::memcpy(ptr, src, static_cast<std::size_t>(sz));
                    vmaUnmapMemory(mAllocator.allocator, b.allocation);
                    };
                up(ps, mesh.positions.data(), posSz);
                up(ts, mesh.texcoords.data(), texSz);
                up(ns, mesh.normals.data(), normSz);
                up(is, mesh.indices.data(), idxSz);

                // vkCmdCopyBuffer
                auto cpy = [&](lut::Buffer& src, lut::Buffer& dst, VkDeviceSize sz) {
                    VkBufferCopy c{ 0, 0, sz };
                    vkCmdCopyBuffer(uploadCmd, src.buffer, dst.buffer, 1, &c);
                    };
                cpy(ps, mMeshPositions.back(), posSz);
                cpy(ts, mMeshTexCoords.back(), texSz);
                cpy(ns, mMeshNormals.back(), normSz);
                cpy(is, mMeshIndices.back(), idxSz);

                // Barriers
                lut::buffer_barrier(uploadCmd, mMeshPositions.back().buffer,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
                lut::buffer_barrier(uploadCmd, mMeshTexCoords.back().buffer,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
                lut::buffer_barrier(uploadCmd, mMeshNormals.back().buffer,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
                lut::buffer_barrier(uploadCmd, mMeshIndices.back().buffer,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT);

                staging.emplace_back(std::move(ps)); staging.emplace_back(std::move(ts));
                staging.emplace_back(std::move(ns)); staging.emplace_back(std::move(is));
            }

            vkEndCommandBuffer(uploadCmd);

            VkCommandBufferSubmitInfo ci{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            ci.commandBuffer = uploadCmd;
            VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &ci;
            vkQueueSubmit2(mWindow.graphicsQueue, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(mWindow.graphicsQueue);
            // staging 离开作用域自动销毁
        }

        // ─────────────────────────────────────────────────────────
        //  Post-process 描述符构建辅助
        // ─────────────────────────────────────────────────────────
        VkDescriptorSet BuildPostDesc(VkImageView imageView, VkBuffer mosaicBuf)
        {
            VkDescriptorSet ds = lut::alloc_desc_set(
                mWindow, mDescPool.handle, mPostLayout.handle);

            VkDescriptorImageInfo ii{};
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii.imageView = imageView;
            ii.sampler = mPostSampler.handle;

            VkDescriptorBufferInfo bi{ mosaicBuf, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet w[2]{};
            w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[0].dstSet = ds; w[0].dstBinding = 0;
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].descriptorCount = 1; w[0].pImageInfo = &ii;

            w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet = ds; w[1].dstBinding = 1;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w[1].descriptorCount = 1; w[1].pBufferInfo = &bi;

            vkUpdateDescriptorSets(mWindow.device, 2, w, 0, nullptr);
            return ds;
        }

        // Resize 后更新 post-proc / vis 描述符里的 image view
        void UpdatePostDescImage(std::vector<VkDescriptorSet>& sets, VkImageView newView)
        {
            VkDescriptorImageInfo ii{};
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ii.imageView = newView;
            ii.sampler = mPostSampler.handle;

            for (auto ds : sets) {
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = ds; w.dstBinding = 0;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.descriptorCount = 1; w.pImageInfo = &ii;
                vkUpdateDescriptorSets(mWindow.device, 1, &w, 0, nullptr);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  成员变量
        // ─────────────────────────────────────────────────────────
        bool& mAppRunning;

        lut::VulkanWindow  mWindow;
        lut::Allocator     mAllocator;

        UserState  mState{};
        bool       mRecreateSwapchain = false;
        std::size_t mFrameIndex = 0;

        lut::CommandPool    mCmdPool;
        lut::DescriptorPool mDescPool;

        std::vector<VkCommandBuffer>  mCmdBuffers;
        std::vector<lut::Fence>       mFrameDone;
        std::vector<lut::Semaphore>   mImageAvailable;
        std::vector<lut::Semaphore>   mRenderFinished;

        // Descriptor layouts / Pipeline layouts
        lut::DescriptorSetLayout mSceneLayout, mObjectLayout, mPostLayout;
        lut::PipelineLayout      mPipeLayout, mPostPipeLayout;

        // Pipelines
        lut::Pipeline mPipe, mAlphaPipe;
        lut::Pipeline mMipPipe, mDepthPipe, mDerivPipe;
        lut::Pipeline mOverdrawPipe, mOvershadingPipe;
        lut::Pipeline mPostProcPipe, mVisResolvePipe;
        lut::Pipeline mShadowPipe;

        // 场景数据（EngineModel，含 scenes 字段）
        EngineModel                    mModel;
        std::vector<lut::Image>        mModelTextures;
        std::vector<lut::ImageView>    mModelTextureViews;

        // 默认灰色纹理（材质缺纹理时占位）
        lut::Image     mDefaultGrayTex;
        lut::ImageView mDefaultGrayView;

        // Samplers
        lut::Sampler mDefaultSampler, mDebugSampler;
        lut::Sampler mPostSampler, mShadowSampler;

        // Mesh GPU buffers
        std::vector<lut::Buffer> mMeshPositions;
        std::vector<lut::Buffer> mMeshTexCoords;
        std::vector<lut::Buffer> mMeshNormals;
        std::vector<lut::Buffer> mMeshIndices;

        // UBOs
        lut::Buffer              mSceneUBO;
        std::vector<lut::Buffer> mMosaicUBOs;

        // Descriptor sets
        VkDescriptorSet                mSceneDescriptors = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet>   mMaterialDescriptors;
        std::vector<VkDescriptorSet>   mDebugMaterialDescriptors;
        std::vector<VkDescriptorSet>   mPostDescriptors;
        std::vector<VkDescriptorSet>   mVisDescriptors;

        // Render targets
        lut::ImageWithView mDepthBuffer;
        lut::ImageWithView mOffscreenImage;
        lut::ImageWithView mVisImage;
        lut::ImageWithView mShadowMap;
    };

} // namespace engine