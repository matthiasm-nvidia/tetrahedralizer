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

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

struct CpuBVH
{
    struct PackedNodeHalf
    {
        float x;
        float y;
        float z;
        unsigned int i : 31;
        unsigned int b : 1;
    };

    int mRootNode = 0;

    std::vector<PackedNodeHalf> mNodeLowers;
    std::vector<PackedNodeHalf> mNodeUppers;

    int mNumNodes = 0;
    int mMaxNodes = 0;
    int mMaxDepth = 0;

    void clear()
    {
        mNodeLowers.clear();
        mNodeUppers.clear();
        mNumNodes = 0;
        mMaxNodes = 0;
        mMaxDepth = 0;
        mRootNode = 0;
    }

    bool empty() const
    {
        return mNodeLowers.empty();
    }

    void query(const Bounds3& bounds, std::vector<int>& items) const;
    void queryRay(const Ray& ray, std::vector<int>& items, float minT = -MaxFloat, float maxT = MaxFloat) const;

    bool raycast(const Ray& ray,
                 const Vec3* verts,
                 const std::uint32_t* triIds,
                 Vec3& bary,
                 int& triNr,
                 float& t,
                 Vec3& normal,
                 bool& inside,
                 const Vec3* clip = nullptr) const;

    bool rayIntersected(const Ray& ray,
                        const std::vector<Vec3>& verts,
                        const std::vector<std::uint32_t>& triIds,
                        float maxT = 0.0f) const;

    bool queryClosestPoint(const Vec3& p,
                           float maxDist,
                           const Vec3* verts,
                           const std::uint32_t* triIds,
                           Vec3& closestBary,
                           int& closestTriNr,
                           Vec3& closestPos,
                           bool& inside) const;

    mutable std::vector<int> mItems;
};

class CpuBVHBuilder
{
public:
    void build(CpuBVH& bvh, const std::vector<Bounds3>& bounds);
    void build(CpuBVH& bvh, const Vec3* verts, const std::uint32_t* triIds, int numTris);
    void build(CpuBVH& bvh, const std::vector<Vec3>& verts, const std::vector<std::uint32_t>& triIds)
    {
        build(bvh, verts.data(), triIds.data(), static_cast<int>(triIds.size() / 3));
    }

private:
    struct KeyIndexPair
    {
        unsigned int key;
        int index;

        inline bool operator<(const KeyIndexPair& rhs) const
        {
            return key < rhs.key;
        }
    };

    Bounds3 CalcBounds(const Bounds3* bounds, const KeyIndexPair* keys, int start, int end);
    int FindSplit(const KeyIndexPair* pairs, int start, int end);
    int BuildRecursive(CpuBVH& bvh, const KeyIndexPair* keys, const Bounds3* bounds, int start, int end, int depth);
};

} // namespace tetrahedralizer
