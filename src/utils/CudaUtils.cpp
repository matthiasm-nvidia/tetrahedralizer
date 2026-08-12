// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "CudaUtils.h"

#include <cstring>

using namespace tetrahedralizer;

cudaTextureObject_t CreateTexture(const Vec4* buffer, int sizeInBytes)
{
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 32;
    resDesc.res.linear.desc.w = 32;
    resDesc.res.linear.sizeInBytes = sizeInBytes;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex;
    cudaCheck(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    return tex;
}

void DestroyTexture(cudaTextureObject_t tex)
{
    if (tex)
        cudaCheck(cudaDestroyTextureObject(tex));
}

void DeviceAlloc(void** ptr, size_t size)
{
    cudaCheck(cudaMalloc(ptr, size));
}

void DeviceFree(void* ptr)
{
    cudaCheck(cudaFree(ptr));
}
