// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "tetrahedralizer/Vec.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace tetrahedralizer
{

struct PackedNodeHalf
{
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;
};

struct GpuBVH
{
    int* mRootNodes = nullptr;
    PackedNodeHalf* __restrict__ mNodeLowers = nullptr;
    PackedNodeHalf* __restrict__ mNodeUppers = nullptr;
    int mNumNodes = 0;
    int mMaxNodes = 0;
    int mNumRoots = 0;
    int mMaxRoots = 0;
    int mMaxDepth = 0;

    void free()
    {
        if (mRootNodes)
        {
            cudaFree(mRootNodes);
            mRootNodes = nullptr;
        }
        if (mNodeLowers)
        {
            cudaFree(mNodeLowers);
            mNodeLowers = nullptr;
        }
        if (mNodeUppers)
        {
            cudaFree(mNodeUppers);
            mNodeUppers = nullptr;
        }
        mMaxNodes = 0;
        mNumNodes = 0;
        mMaxRoots = 0;
        mNumRoots = 0;
        mMaxDepth = 0;
    }
};

struct BVHBuilderDeviceData;

class BVHBuilderGPU
{
public:
    BVHBuilderGPU() = default;
    ~BVHBuilderGPU() = default;

    void build(GpuBVH&, const Vec4*, const Vec4*, int, const int* = nullptr, int = 1) {}
    static void resizeBVH(GpuBVH&, int, int = 1) {}
    static void freeBVH(GpuBVH& bvh) { bvh.free(); }
    static void cloneBVH(const GpuBVH&, GpuBVH&) {}
    void free() {}
    size_t allocationSize() { return 0; }
};

} // namespace tetrahedralizer
