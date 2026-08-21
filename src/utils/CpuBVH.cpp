// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
#include "utils/CpuBVH.h"

#include "utils/Geometry.h"

#include <algorithm>

namespace tetrahedralizer
{

void CpuBVH::query(const Bounds3& bounds, std::vector<int>& items) const
{
    items.clear();

    if (empty())
        return;

    int stack[64];
    stack[0] = mRootNode;
    int count = 1;

    while (count)
    {
        const int nodeIndex = stack[--count];

        const PackedNodeHalf& lower = mNodeLowers[static_cast<std::size_t>(nodeIndex)];
        const PackedNodeHalf& upper = mNodeUppers[static_cast<std::size_t>(nodeIndex)];

        if (Bounds3(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z)).intersect(bounds))
        {
            const int leftIndex = static_cast<int>(lower.i);
            const int rightIndex = static_cast<int>(upper.i);

            if (lower.b)
                items.push_back(leftIndex);
            else
            {
                stack[count++] = leftIndex;
                stack[count++] = rightIndex;
            }
        }
    }
}

void CpuBVH::queryRay(const Ray& ray, std::vector<int>& items, float minT, float maxT) const
{
    items.clear();

    if (empty())
        return;

    int stack[64];
    stack[0] = mRootNode;
    int count = 1;
    float tEntry = 0.0f;
    float tExit = 0.0f;

    while (count)
    {
        const int nodeIndex = stack[--count];

        const PackedNodeHalf& lower = mNodeLowers[static_cast<std::size_t>(nodeIndex)];
        const PackedNodeHalf& upper = mNodeUppers[static_cast<std::size_t>(nodeIndex)];
        const Bounds3 bounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (header_rayBoundsIntersection(ray, bounds, &tEntry, &tExit))
        {
            if (tExit < minT || tEntry > maxT)
                continue;

            const int leftIndex = static_cast<int>(lower.i);
            const int rightIndex = static_cast<int>(upper.i);

            if (lower.b)
                items.push_back(leftIndex);
            else
            {
                stack[count++] = leftIndex;
                stack[count++] = rightIndex;
            }
        }
    }
}

bool CpuBVH::rayIntersected(const Ray& ray,
                            const std::vector<Vec3>& verts,
                            const std::vector<std::uint32_t>& triIds,
                            float maxT) const
{
    if (empty())
        return false;

    queryRay(ray, mItems);

    for (int rayTriNr : mItems)
    {
        const Vec3& p0 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr)]];
        const Vec3& p1 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr + 1)]];
        const Vec3& p2 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr + 2)]];

        if (ray.orig == p0 || ray.orig == p1 || ray.orig == p2)
            continue;

        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (header_rayTriangleIntersection(ray, p0, p1, p2, t, u, v) && t >= 0.0f &&
            (t <= maxT || maxT == 0.0f))
            return true;
    }

    return false;
}

bool CpuBVH::raycast(const Ray& ray,
                     const Vec3* verts,
                     const std::uint32_t* triIds,
                     Vec3& bary,
                     int& triNr,
                     float& t,
                     Vec3& normal,
                     bool& inside,
                     const Vec3* clip) const
{
    if (empty())
        return false;

    queryRay(ray, mItems);

    float minT = 0.0f;
    triNr = -1;

    for (int rayTriNr : mItems)
    {
        const Vec3& p0 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr)]];
        const Vec3& p1 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr + 1)]];
        const Vec3& p2 = verts[triIds[static_cast<std::size_t>(3 * rayTriNr + 2)]];

        if (ray.orig == p0 || ray.orig == p1 || ray.orig == p2)
            continue;

        float ti = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (!header_rayTriangleIntersection(ray, p0, p1, p2, ti, u, v) || !(ti >= 0.0f))
            continue;

        const Vec3 hit = ray.at(ti);
        if (clip && !(hit.x < clip->x && hit.y < clip->y && hit.z < clip->z))
            continue;

        if (triNr < 0 || ti < minT)
        {
            minT = ti;
            triNr = rayTriNr;
            bary = Vec3(1.0f - u, u, v);
            normal = (p1 - p0).cross(p2 - p0).normalized();
            inside = (p1 - p0).cross(p2 - p0).dot(ray.dir) > 0.0f;
        }
    }

    t = minT;
    return triNr >= 0;
}

bool CpuBVH::queryClosestPoint(const Vec3& p,
                               float maxDist,
                               const Vec3* verts,
                               const std::uint32_t* triIds,
                               Vec3& closestBary,
                               int& closestTriNr,
                               Vec3& closestPos,
                               bool& inside) const
{
    if (empty())
        return false;

    int stack[64];
    stack[0] = mRootNode;
    int count = 1;

    float minDist2 = maxDist > 0.0f ? maxDist * maxDist : MaxFloat;
    int minTriNr = -1;
    Vec3 minBary(Zero);
    bool minInside = false;
    Vec3 minPos(Zero);

    while (count)
    {
        const int nodeIndex = stack[--count];

        const PackedNodeHalf& lower = mNodeLowers[static_cast<std::size_t>(nodeIndex)];
        const PackedNodeHalf& upper = mNodeUppers[static_cast<std::size_t>(nodeIndex)];

        const float nodeDist2 =
            header_distanceToBoundsSquared(p, Bounds3(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z)));
        if (nodeDist2 > minDist2)
            continue;

        const int leftIndex = static_cast<int>(lower.i);
        const int rightIndex = static_cast<int>(upper.i);

        if (lower.b)
        {
            const int i = static_cast<int>(triIds[static_cast<std::size_t>(leftIndex * 3 + 0)]);
            const int j = static_cast<int>(triIds[static_cast<std::size_t>(leftIndex * 3 + 1)]);
            const int k = static_cast<int>(triIds[static_cast<std::size_t>(leftIndex * 3 + 2)]);

            const Vec3 p0 = verts[i];
            const Vec3 p1 = verts[j];
            const Vec3 p2 = verts[k];

            const Vec3 e0 = p1 - p0;
            const Vec3 e1 = p2 - p0;
            const Vec3 e2 = p2 - p1;
            const Vec3 normal = e0.cross(e1);

            if (normal.magnitude() / (e0.dot(e0) + e1.dot(e1) + e2.dot(e2)) < 1.e-6f)
                continue;

            const Vec3 bary = header_GetClosestPointOnTriangle(p, p0, p1, p2);
            const Vec3 c = bary.x * p0 + bary.y * p1 + bary.z * p2;
            const float dist2 = (c - p).magnitudeSquared();

            if (dist2 < minDist2)
            {
                minDist2 = dist2;
                minBary = bary;
                minTriNr = leftIndex;
                minInside = (c - p).dot(normal) > 0.0f;
                minPos = c;
            }
        }
        else
        {
            const PackedNodeHalf& leftLower = mNodeLowers[static_cast<std::size_t>(leftIndex)];
            const PackedNodeHalf& leftUpper = mNodeUppers[static_cast<std::size_t>(leftIndex)];
            const PackedNodeHalf& rightLower = mNodeLowers[static_cast<std::size_t>(rightIndex)];
            const PackedNodeHalf& rightUpper = mNodeUppers[static_cast<std::size_t>(rightIndex)];

            const float leftDist2 = header_distanceToBoundsSquared(
                p, Bounds3(Vec3(leftLower.x, leftLower.y, leftLower.z), Vec3(leftUpper.x, leftUpper.y, leftUpper.z)));
            const float rightDist2 = header_distanceToBoundsSquared(
                p,
                Bounds3(Vec3(rightLower.x, rightLower.y, rightLower.z), Vec3(rightUpper.x, rightUpper.y, rightUpper.z)));

            if (leftDist2 < rightDist2)
            {
                if (rightDist2 < minDist2)
                    stack[count++] = rightIndex;
                if (leftDist2 < minDist2)
                    stack[count++] = leftIndex;
            }
            else
            {
                if (leftDist2 < minDist2)
                    stack[count++] = leftIndex;
                if (rightDist2 < minDist2)
                    stack[count++] = rightIndex;
            }
        }
    }

    if (maxDist == 0.0f || minDist2 < maxDist * maxDist)
    {
        closestBary = minBary;
        closestTriNr = minTriNr;
        inside = minInside;
        closestPos = minPos;
        return true;
    }

    return false;
}

inline int CLZ(int x)
{
    int n = 0;
    if (x == 0)
        return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1)
        ;
    return n;
}

inline unsigned int Part1by2(unsigned int n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    return n;
}

template <int Dim>
inline unsigned int Morton3(float x, float y, float z)
{
    const unsigned int ux = static_cast<unsigned int>(Clamp(int(x * Dim), 0, Dim - 1));
    const unsigned int uy = static_cast<unsigned int>(Clamp(int(y * Dim), 0, Dim - 1));
    const unsigned int uz = static_cast<unsigned int>(Clamp(int(z * Dim), 0, Dim - 1));
    return (Part1by2(uz) << 2) | (Part1by2(uy) << 1) | Part1by2(ux);
}

void CpuBVHBuilder::build(CpuBVH& bvh, const std::vector<Bounds3>& bounds)
{
    const int n = static_cast<int>(bounds.size());

    bvh.mMaxDepth = 0;
    bvh.mMaxNodes = 2 * n;
    bvh.mNodeLowers.resize(static_cast<std::size_t>(bvh.mMaxNodes));
    bvh.mNodeUppers.resize(static_cast<std::size_t>(bvh.mMaxNodes));
    bvh.mNumNodes = 0;
    bvh.mRootNode = 0;

    if (n <= 0)
    {
        bvh.clear();
        return;
    }

    std::vector<KeyIndexPair> keys;
    keys.reserve(static_cast<std::size_t>(n));

    Bounds3 totalBounds(Empty);
    for (int i = 0; i < n; ++i)
        totalBounds.include(bounds[static_cast<std::size_t>(i)]);
    totalBounds.expand(0.001f);

    const Vec3 edges = totalBounds.getDimensions();
    const Vec3 invEdges(1.0f / edges.x, 1.0f / edges.y, 1.0f / edges.z);

    for (int i = 0; i < n; ++i)
    {
        const Vec3 center = bounds[static_cast<std::size_t>(i)].getCenter();
        const Vec3 local = (center - totalBounds.minimum).multiply(invEdges);

        KeyIndexPair pair;
        pair.key = Morton3<1024>(local.x, local.y, local.z);
        pair.index = i;
        keys.push_back(pair);
    }

    std::sort(keys.begin(), keys.end());
    BuildRecursive(bvh, keys.data(), bounds.data(), 0, n, 0);
}

void CpuBVHBuilder::build(CpuBVH& bvh, const Vec3* verts, const std::uint32_t* triIds, int numTris)
{
    if (verts == nullptr || triIds == nullptr || numTris <= 0)
    {
        bvh.clear();
        return;
    }

    std::vector<Bounds3> bounds(static_cast<std::size_t>(numTris));
    for (int i = 0; i < numTris; i++)
    {
        const Vec3& p0 = verts[triIds[static_cast<std::size_t>(3 * i)]];
        const Vec3& p1 = verts[triIds[static_cast<std::size_t>(3 * i + 1)]];
        const Vec3& p2 = verts[triIds[static_cast<std::size_t>(3 * i + 2)]];

        Bounds3& box = bounds[static_cast<std::size_t>(i)];
        box.minimum = p0;
        box.maximum = p0;
        box.include(p1);
        box.include(p2);
        box.expand(1.0e-5f);
    }

    build(bvh, bounds);
}

Bounds3 CpuBVHBuilder::CalcBounds(const Bounds3* bounds, const KeyIndexPair* keys, int start, int end)
{
    Bounds3 u(Empty);
    for (int i = start; i < end; ++i)
        u.include(bounds[keys[i].index]);
    return u;
}

int CpuBVHBuilder::FindSplit(const KeyIndexPair* pairs, int start, int end)
{
    if (pairs[start].key == pairs[end - 1].key)
        return (start + end) / 2;

    const int commonPrefix = CLZ(static_cast<int>(pairs[start].key ^ pairs[end - 1].key));
    const int mask = 1 << (31 - commonPrefix);

    while (end - start > 0)
    {
        const int index = (start + end) / 2;
        if (pairs[index].key & static_cast<unsigned int>(mask))
            end = index;
        else
            start = index + 1;
    }

    return start;
}

CpuBVH::PackedNodeHalf MakeNode(const Vec3& bound, int child, bool leaf)
{
    CpuBVH::PackedNodeHalf n;
    n.x = bound.x;
    n.y = bound.y;
    n.z = bound.z;
    n.i = static_cast<unsigned int>(child) & 0x7FFFFFFF;
    n.b = leaf ? 1u : 0u;
    return n;
}

int CpuBVHBuilder::BuildRecursive(CpuBVH& bvh, const KeyIndexPair* keys, const Bounds3* bounds, int start, int end,
                                  int depth)
{
    const int n = end - start;
    const int nodeIndex = bvh.mNumNodes++;

    if (depth > bvh.mMaxDepth)
        bvh.mMaxDepth = depth;

    const Bounds3 box = CalcBounds(bounds, keys, start, end);
    constexpr int kMaxItemsPerLeaf = 1;

    if (n <= kMaxItemsPerLeaf)
    {
        bvh.mNodeLowers[static_cast<std::size_t>(nodeIndex)] = MakeNode(box.minimum, keys[start].index, true);
        bvh.mNodeUppers[static_cast<std::size_t>(nodeIndex)] = MakeNode(box.maximum, keys[start].index, false);
    }
    else
    {
        int split = FindSplit(keys, start, end);
        if (split <= start || split >= end)
            split = (start + end) / 2;
        const int leftChild = BuildRecursive(bvh, keys, bounds, start, split, depth + 1);
        const int rightChild = BuildRecursive(bvh, keys, bounds, split, end, depth + 1);
        bvh.mNodeLowers[static_cast<std::size_t>(nodeIndex)] = MakeNode(box.minimum, leftChild, false);
        bvh.mNodeUppers[static_cast<std::size_t>(nodeIndex)] = MakeNode(box.maximum, rightChild, false);
    }

    return nodeIndex;
}

} // namespace tetrahedralizer
