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
    GpuBVH() : mRootNodes(nullptr), mNodeLowers(nullptr), mNodeUppers(nullptr), mNumNodes(0), mMaxNodes(0), mMaxDepth(0)
    {
    }

    // The entries hold the indices of the roots of the BVH trees
    // If no groups are used, there is a single root node
    // otherwise one root node per group
    int* mRootNodes;

    PackedNodeHalf* __restrict__ mNodeLowers; // x, y, z are the lower spatial bounds of the node's children
                                              // for internal nodes b is zero and i is a pointher the the left child
                                              // node for leaf nodes b is non zero and i is the *global* number of the
                                              // item

    PackedNodeHalf* __restrict__ mNodeUppers; // x, y, z are the upper spatial bounds of the node's children
                                              // for internal nodes i is a pointer to the right child node
                                              // for leaf nodes i is the *local* number of the item with respect to the
                                              // group start b is not used

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


    size_t size() const
    {
        return sizeof(PackedNodeHalf) * mMaxNodes * 2 + mNumRoots * sizeof(int);
    }
};

//

/////////////////////////////////////////////////////////////////////////////////////////////
// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//


struct BVHBuilderDeviceData;

class BVHBuilderGPU
{
public:
    BVHBuilderGPU();
    ~BVHBuilderGPU();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each item
    // for multiple groups, firstItemOfGroup defines the position of the first item of each group,
    // if nullptr, a single group is assumed

    void build(GpuBVH& bvh,
               const Vec4* itemLowers,
               const Vec4* itemUppers,
               int numItems,
               const int* firstItemOfGroup = nullptr,
               int numGroups = 1);

    static void resizeBVH(GpuBVH& bvh, int numNodes, int numRoots = 1);
    static void freeBVH(GpuBVH& bvh);
    static void cloneBVH(const GpuBVH& hostBVH, GpuBVH& deviceBVH);
    void free();
    size_t allocationSize();


private:
    // temporary data used during building

    BVHBuilderDeviceData* mDeviceData = nullptr;
};

} // namespace tetrahedralizer
