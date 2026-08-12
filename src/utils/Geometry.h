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

#include "utils/Math.h"
#include "utils/CudaUtils.h"
#include "tetrahedralizer/Vec.h"

namespace tetrahedralizer
{


// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline float header_minMax(float f0, float f1, float f2)
{
    return Max(-Max(f0, f1, f2), Min(f0, f1, f2));
}

CUDA_CALLABLE inline bool header_boxTriangleIntersection(
    const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& center, const Vec3& extents)
{
    Vec3 v0 = p0 - center, v1 = p1 - center, v2 = p2 - center;
    Vec3 f0 = p1 - p0, f1 = p2 - p1, f2 = p0 - p2;
    float r;

    Vec3 n = f0.cross(f1);
    float d = n.dot(v0);
    r = extents.x * fabsf(n.x) + extents.y * fabsf(n.y) + extents.z * fabsf(n.z);
    if (d > r || d < -r)
        return false;

    if (Max(v0.x, v1.x, v2.x) < -extents.x || Min(v0.x, v1.x, v2.x) > extents.x)
        return false;

    if (Max(v0.y, v1.y, v2.y) < -extents.y || Min(v0.y, v1.y, v2.y) > extents.y)
        return false;

    if (Max(v0.z, v1.z, v2.z) < -extents.z || Min(v0.z, v1.z, v2.z) > extents.z)
        return false;

    Vec3 a00(0.0f, -f0.z, f0.y);
    r = extents.y * fabsf(f0.z) + extents.z * fabsf(f0.y);
    if (header_minMax(v0.dot(a00), v1.dot(a00), v2.dot(a00)) > r)
        return false;

    Vec3 a01(0.0f, -f1.z, f1.y);
    r = extents.y * fabsf(f1.z) + extents.z * fabsf(f1.y);
    if (header_minMax(v0.dot(a01), v1.dot(a01), v2.dot(a01)) > r)
        return false;

    Vec3 a02(0.0f, -f2.z, f2.y);
    r = extents.y * fabsf(f2.z) + extents.z * fabsf(f2.y);
    if (header_minMax(v0.dot(a02), v1.dot(a02), v2.dot(a02)) > r)
        return false;

    Vec3 a10(f0.z, 0.0f, -f0.x);
    r = extents.x * fabsf(f0.z) + extents.z * fabsf(f0.x);
    if (header_minMax(v0.dot(a10), v1.dot(a10), v2.dot(a10)) > r)
        return false;

    Vec3 a11(f1.z, 0.0f, -f1.x);
    r = extents.x * fabsf(f1.z) + extents.z * fabsf(f1.x);
    if (header_minMax(v0.dot(a11), v1.dot(a11), v2.dot(a11)) > r)
        return false;

    Vec3 a12(f2.z, 0.0f, -f2.x);
    r = extents.x * fabsf(f2.z) + extents.z * fabsf(f2.x);
    if (header_minMax(v0.dot(a12), v1.dot(a12), v2.dot(a12)) > r)
        return false;

    Vec3 a20(-f0.y, f0.x, 0.0f);
    r = extents.x * fabsf(f0.y) + extents.y * fabsf(f0.x);
    if (header_minMax(v0.dot(a20), v1.dot(a20), v2.dot(a20)) > r)
        return false;

    Vec3 a21(-f1.y, f1.x, 0.0f);
    r = extents.x * fabsf(f1.y) + extents.y * fabsf(f1.x);
    if (header_minMax(v0.dot(a21), v1.dot(a21), v2.dot(a21)) > r)
        return false;

    Vec3 a22(-f2.y, f2.x, 0.0f);
    r = extents.x * fabsf(f2.y) + extents.y * fabsf(f2.x);
    if (header_minMax(v0.dot(a22), v1.dot(a22), v2.dot(a22)) > r)
        return false;

    return true;
}


//-----------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_rayBoundsIntersection(Ray ray,
                                                       const Bounds3 bounds,
                                                       float* rayEntry = nullptr,
                                                       float* rayExit = nullptr)
{
    float tEntry = -MaxFloat;
    float tExit = MaxFloat;

    if (rayEntry && rayExit)
    {
        *rayEntry = tEntry;
        *rayExit = tExit;
    }

    for (int i = 0; i < 3; ++i)
    {
        if (ray.dir[i] != 0.0f)
        {
            float t1 = (bounds.minimum[i] - ray.orig[i]) / ray.dir[i];
            float t2 = (bounds.maximum[i] - ray.orig[i]) / ray.dir[i];

            tEntry = Max(tEntry, Min(t1, t2));
            tExit = Min(tExit, Max(t1, t2));
        }
        else if (ray.orig[i] < bounds.minimum[i] || ray.orig[i] > bounds.maximum[i])
            return false;
    }

    if (rayEntry && rayExit)
    {
        *rayEntry = tEntry;
        *rayExit = tExit;
    }

    return tExit > 0.0f && tEntry < tExit;
}


//-----------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_rayTriangleIntersection(
    const Ray& ray, const Vec3& a, const Vec3& b, const Vec3& c, float& t, float& u, float& v)
{
    t = MaxFloat;

    Vec3 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det;

    edge1 = b - a;
    edge2 = c - a;
    pvec = ray.dir.cross(edge2);
    det = edge1.dot(pvec);

    if (det == 0.0f)
        return false;
    inv_det = 1.0f / det;
    tvec = ray.orig - a;

    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return false;

    qvec = tvec.cross(edge1);
    v = ray.dir.dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = edge2.dot(qvec) * inv_det;

    return true;
}

// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_triangleTriangleIntersection(
    const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& q0, const Vec3& q1, const Vec3& q2, Vec3& s0, Vec3& s1)
{
    float t, u, v;
    bool intersects = false;
    Vec3 p[3] = { p0, p1, p2 };
    Vec3 q[3] = { q0, q1, q2 };
    Vec3 cuts[2];
    int numCuts = 0;

    for (int i = 0; i < 3; i++)
    {
        int j = (i + 1) % 3;
        intersects = header_rayTriangleIntersection(Ray(p[i], p[j] - p[i]), q0, q1, q2, t, u, v);
        if (intersects && 0.0f <= t && t <= 1.0f)
        {
            cuts[numCuts++] = p[i] + (p[j] - p[i]) * t;
            if (numCuts == 2)
                break;
        }
        intersects = header_rayTriangleIntersection(Ray(q[i], q[j] - q[i]), p0, p1, p2, t, u, v);
        if (intersects && 0.0f <= t && t <= 1.0f)
        {
            cuts[numCuts++] = q[i] + (q[j] - q[i]) * t;
            if (numCuts == 2)
                break;
        }
    }
    s0 = cuts[0];
    s1 = cuts[1];
    return numCuts == 2;
}


// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_raycast(GpuBVH bvh,
                                         Vec3* vertices,
                                         int* triIds,
                                         const Ray& ray,
                                         float& minT,
                                         int& minTriNr,
                                         int* excludeVertNr = nullptr,
                                         int* excludeTriNr = nullptr)
{

    int stack[64];
    stack[0] = bvh.mRootNodes[0];
    int count = 1;

    minT = 0.0f;
    minTriNr = -1;

    // traverse bvh using ray

    while (count)
    {
        const int nodeIndex = stack[--count];

        PackedNodeHalf lower = bvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = bvh.mNodeUppers[nodeIndex];


        Bounds3 bounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (header_rayBoundsIntersection(ray, bounds))
        {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b)
            {
                int triNr = leftIndex;

                if (excludeTriNr)
                {
                    if (triNr == *excludeTriNr)
                    {
                        continue;
                    }
                }

                int id0 = triIds[3 * triNr + 0];
                int id1 = triIds[3 * triNr + 1];
                int id2 = triIds[3 * triNr + 2];

                if (excludeVertNr)
                {
                    if (id0 == *excludeVertNr || id1 == *excludeVertNr || id2 == *excludeVertNr)
                    {
                        continue;
                    }
                }

                Vec3 p0 = vertices[id0];
                Vec3 p1 = vertices[id1];
                Vec3 p2 = vertices[id2];

                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;

                bool hit = header_rayTriangleIntersection(ray, p0, p1, p2, t, u, v);

                if (hit && t >= 0.0f && (minTriNr < 0 || t < minT))
                {
                    minT = t;
                    minTriNr = triNr;
                }
            }
            else
            {
                stack[count++] = leftIndex;
                stack[count++] = rightIndex;
            }
        }
    }
    return minTriNr >= 0;
}


// ----------------------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline Vec3 header_getClosestPointOnBounds(const Vec3& p, const Bounds3& bounds)
{
    Vec3 c = p;

    if (c.x < bounds.minimum.x)
        c.x = bounds.minimum.x;
    if (c.x > bounds.maximum.x)
        c.x = bounds.maximum.x;

    if (c.y < bounds.minimum.y)
        c.y = bounds.minimum.y;
    if (c.y > bounds.maximum.y)
        c.y = bounds.maximum.y;

    if (c.z < bounds.minimum.z)
        c.z = bounds.minimum.z;
    if (c.z > bounds.maximum.z)
        c.z = bounds.maximum.z;

    return c;
}

// ----------------------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline float header_distanceToBoundsSquared(const Vec3& p, const Bounds3& bounds)
{
    return (p - header_getClosestPointOnBounds(p, bounds)).magnitudeSquared();
}

CUDA_CALLABLE inline Vec3 header_GetClosestPointOnTriangle(
    const Vec3& p, const Vec3& p0, const Vec3& p1, const Vec3& p2, bool* inside = nullptr)
{
    Vec3 e0 = p1 - p0;
    Vec3 e1 = p2 - p0;
    Vec3 tmp = p0 - p;

    float a = e0.dot(e0);
    float b = e0.dot(e1);
    float c = e1.dot(e1);
    float d = e0.dot(tmp);
    float e = e1.dot(tmp);
    Vec3 coords(b * e - c * d, b * d - a * e, a * c - b * b);

    float x = 0.0f;
    float y = 0.0f;
    if (inside)
    {
        *inside = false;
    }
    if (coords[0] <= 0.0f)
    {
        if (c != 0.0f)
            y = -e / c;
    }
    else if (coords[1] <= 0.0f)
    {
        if (a != 0.0f)
            x = -d / a;
    }
    else if (coords[0] + coords[1] > coords[2])
    {
        float den = a + c - b - b;
        float num = c + e - b - d;
        if (den != 0.0f)
        {
            x = num / den;
            y = 1.0f - x;
        }
    }
    else
    {
        if (coords[2] != 0.0f)
        {
            x = coords[0] / coords[2];
            y = coords[1] / coords[2];
        }
        if (inside)
        {
            *inside = true;
        }
    }

    x = Clamp(x, 0.0f, 1.0f);
    y = Clamp(y, 0.0f, 1.0f);

    return Vec3(1.0f - x - y, x, y);
}


// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline float header_getClosestPointOnRay(const Vec3& p, const Ray& ray)
{
    if (ray.dir == Vec3(Zero))
    {
        return 0.0f;
    }
    return ray.dir.dot(p - ray.orig) / ray.dir.magnitudeSquared();
}


// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline Vec3 header_getClosestPointOnSegment(
    const Vec3& p, const Vec3& p0, const Vec3& p1, bool* inside = nullptr, float* segmentT = nullptr)
{
    float t = 0.0f;
    bool isInside = false;

    Vec3 d = p1 - p0;
    float d2 = d.dot(d);

    if (d2 == 0.0f) // degenerate segment
    {
        isInside = (p == p0);
        t = 0.0f;
    }
    else
    {
        t = d.dot(p - p0) / d2;
        if (t < 0.0f)
        {
            t = 0.0f;
        }
        else if (t > 1.0f)
        {
            t = 1.0f;
        }
        else
        {
            isInside = true;
        }
    }

    if (inside)
    {
        *inside = isInside;
    }
    if (segmentT)
    {
        *segmentT = t;
    }

    return p0 + d * t;
}

// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_getClosestPointsOnRays(const Ray& ray0, const Ray& ray1, float& t0, float& t1)
{
    float a = ray0.dir.magnitudeSquared();
    float b = -ray0.dir.dot(ray1.dir);
    float c = ray0.dir.dot(ray1.dir);
    float d = -ray1.dir.magnitudeSquared();
    float e = (ray1.orig - ray0.orig).dot(ray0.dir);
    float f = (ray1.orig - ray0.orig).dot(ray1.dir);
    float det = a * d - b * c;
    if (det == 0.0f) // rays are parallel
    {
        t0 = 0.0f; // arbitrary
        t1 = 0.0f;
        return false;
    }
    det = 1.0f / det;
    t0 = (e * d - b * f) * det;
    t1 = (a * f - e * c) * det;
    return true;
}

// -----------------------------------------------------------------------------------
CUDA_CALLABLE inline bool header_queryClosestPoint(GpuBVH bvh,
                                                   Vec3& p,
                                                   float maxDist,
                                                   Vec3* vertices,
                                                   int* triIds,
                                                   int& closestTriNr,
                                                   Vec3& baryCoords,
                                                   Vec3& closestPos,
                                                   bool& inside)
{

    int stack[64];
    stack[0] = bvh.mRootNodes[0];
    int count = 1;

    float minDist2 = maxDist > 0.0f ? maxDist * maxDist : MaxFloat;
    int minTriNr = -1;
    Vec3 minBary(Zero);
    bool minInside = false;
    Vec3 minPos(Zero);

    while (count)
    {
        const int nodeIndex = stack[--count];

        // union
        //{
        //     PackedNodeHalf lower;
        //     float4 lowerf;
        // };
        // union
        //{
        //     PackedNodeHalf upper;
        //     float4 upperf;
        // };

        // lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
        // upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

        PackedNodeHalf lowerTest = bvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upperTest = bvh.mNodeUppers[nodeIndex];


        // re-test distance
        float nodeDist2 = header_distanceToBoundsSquared(
            p, Bounds3(Vec3(lowerTest.x, lowerTest.y, lowerTest.z), Vec3(upperTest.x, upperTest.y, upperTest.z)));

        if (nodeDist2 > minDist2)
        {
            continue;
        }

        const int leftIndex = lowerTest.i;
        const int rightIndex = upperTest.i;

        if (lowerTest.b)
        {
            // compute closest point on tri
            int i = triIds[leftIndex * 3 + 0];
            int j = triIds[leftIndex * 3 + 1];
            int k = triIds[leftIndex * 3 + 2];

            Vec3 p0 = vertices[i];
            Vec3 p1 = vertices[j];
            Vec3 p2 = vertices[k];

            Vec3 e0 = p1 - p0;
            Vec3 e1 = p2 - p0;
            Vec3 e2 = p2 - p1;
            Vec3 normal = e0.cross(e1);

            // sliver detection
            if (normal.magnitude() / (e0.dot(e0) + e1.dot(e1) + e2.dot(e2)) < 1.e-6f)
                continue;

            bool insideTriangle = false;

            Vec3 bary = header_GetClosestPointOnTriangle(p, p0, p1, p2, &insideTriangle);

            Vec3 c = bary.x * p0 + bary.y * p1 + bary.z * p2;

            float dist2 = (c - p).magnitudeSquared();

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
            // union
            //{
            //     PackedNodeHalf leftLower;
            //     float4 leftLowerf;
            // };
            // union
            //{
            //     PackedNodeHalf leftUpper;
            //     float4 leftUpperf;
            // };

            // leftLowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, leftIndex);
            // leftUpperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, leftIndex);

            PackedNodeHalf leftLowerTest = bvh.mNodeLowers[leftIndex];
            PackedNodeHalf leftUpperTest = bvh.mNodeUppers[leftIndex];

            // union
            //{
            //     PackedNodeHalf rightLower;
            //     float4 rightLowerf;
            // };
            // union
            //{
            //     PackedNodeHalf rightUpper;
            //     float4 rightUpperf;
            // };

            // rightLowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, rightIndex);
            // rightUpperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, rightIndex);

            PackedNodeHalf rightLowerTest = bvh.mNodeLowers[rightIndex];
            PackedNodeHalf rightUpperTest = bvh.mNodeUppers[rightIndex];

            float leftDist2 =
                header_distanceToBoundsSquared(p, Bounds3(Vec3(leftLowerTest.x, leftLowerTest.y, leftLowerTest.z),
                                                          Vec3(leftUpperTest.x, leftUpperTest.y, leftUpperTest.z)));
            float rightDist2 =
                header_distanceToBoundsSquared(p, Bounds3(Vec3(rightLowerTest.x, rightLowerTest.y, rightLowerTest.z),
                                                          Vec3(rightUpperTest.x, rightUpperTest.y, rightUpperTest.z)));

            float leftScore = leftDist2;
            float rightScore = rightDist2;

            if (leftScore < rightScore)
            {
                // put left on top of the stack
                if (rightDist2 < minDist2)
                    stack[count++] = rightIndex;

                if (leftDist2 < minDist2)
                    stack[count++] = leftIndex;
            }
            else
            {
                // put right on top of the stack
                if (leftDist2 < minDist2)
                    stack[count++] = leftIndex;

                if (rightDist2 < minDist2)
                    stack[count++] = rightIndex;
            }
        }
    }

    // check if we found a point, and write outputs
    if (maxDist == 0.0f || minDist2 < maxDist * maxDist)
    {
        baryCoords = minBary;
        closestTriNr = minTriNr;
        inside = minInside;
        closestPos = minPos;

        return true;
    }
    else
    {
        return false;
    }
}


CUDA_CALLABLE inline Vec3 header_getCircumCenterOfTriangle(const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
    Vec3 q1 = p1 - p0;
    Vec3 q2 = p2 - p0;
    Vec3 c = q1.cross(q2);

    float c2 = c.magnitudeSquared();
    if (c2 == 0.0f)
    {
        return (p0 + p1 + p2) / 3.0f;
    }
    Vec3 m = (c.cross(q1) * q2.magnitudeSquared() + q2.cross(c) * q1.magnitudeSquared()) / (2.0f * c2);

    return p0 + m;
}


CUDA_CALLABLE inline float header_computeEdgeCurvature(const Vec3& e0, const Vec3& e1, const Vec3& p0, const Vec3& p1)
{
    if ((e1 - e0).cross(p0 - e0).dot(p1 - e0) == 0.0f)
    {
        return 0.0f; // the points are co-planar -> the curvature is zero
    }

    Vec3 d = e1 - e0;
    float d2 = d.dot(d);
    float t0 = d.dot(p0 - e0) / d2;
    float t1 = d.dot(p1 - e0) / d2;

    Vec3 d0 = e0 + d * t0 - p0;
    Vec3 d1 = e0 + d * t1 - p1;

    Vec3 center = header_getCircumCenterOfTriangle(e0, e0 + d0, e0 + d1);
    float radius = (e0 - center).length();

    return radius != 0.0f ? 1.0f / radius : 0.0f;
}


CUDA_CALLABLE inline Vec3 header_getBarycentricCoordinates(const Vec3& p, const Vec3& p0, const Vec3& p1, const Vec3& p2)
{
    Vec3 e0 = p1 - p0;
    Vec3 e1 = p2 - p0;
    Vec3 tmp = p0 - p;

    float a = e0.dot(e0);
    float b = e0.dot(e1);
    float c = e1.dot(e1);
    float d = e0.dot(tmp);
    float e = e1.dot(tmp);
    float det = a * c - b * b;
    float s = det != 0.0f ? (b * e - c * d) / det : 0.0f;
    float t = det != 0.0f ? (b * d - a * e) / det : 0.0f;
    return Vec3(1.0f - s - t, s, t);
}


CUDA_CALLABLE inline OBB header_computeOBB(Vec3* vertices,
                                           int numVertices,
                                           int* faceIds,
                                           int* faceSizes,
                                           int numFaces,
                                           bool fromSurface,
                                           int* sign = nullptr,
                                           float* surfaceArea = nullptr)
{
    OBB obb = OBB(Empty);
    Vec3 center(Zero);
    Mat33 A(Zero);
    if (surfaceArea)
    {
        *surfaceArea = 0.0f;
    }

    if (fromSurface)
    {
        if (numFaces == 0)
        {
            return obb;
        }

        float areaSum = 0.0f;

        int firstId = 0;

        for (size_t i = 0; i < (size_t)numFaces; i++)
        {
            int faceSize = faceSizes[i];

            for (int j = 1; j < faceSize - 1; j++)
            {
                const Vec3& p0 = vertices[faceIds[(size_t)firstId]];
                const Vec3& p1 = vertices[faceIds[(size_t)firstId + j]];
                const Vec3& p2 = vertices[faceIds[(size_t)firstId + j + 1]];

                float area = 0.5f * (p1 - p0).cross(p2 - p0).magnitude();
                center += (p0 + p1 + p2) / 3.0f * area;
                areaSum += area;
            }

            firstId += faceSize;
        }

        if (surfaceArea)
        {
            *surfaceArea = areaSum;
        }

        center /= areaSum;

        firstId = 0;

        for (size_t i = 0; i < (size_t)numFaces; i++)
        {
            int faceSize = faceSizes[i];

            for (int j = 1; j < faceSize - 1; j++)
            {
                const Vec3& p0 = vertices[faceIds[(size_t)firstId]];
                const Vec3& p1 = vertices[faceIds[(size_t)firstId + j]];
                const Vec3& p2 = vertices[faceIds[(size_t)firstId + j + 1]];

                float area = 0.5f * (p1 - p0).cross(p2 - p0).magnitude();

                const Vec3 r = (p0 + p1 + p2) / 3.0f - center;
                A += area * outerProduct(r, r);
            }

            firstId += faceSize;
        }

        A /= areaSum;
    }
    else
    {
        if (numVertices == 0)
        {
            return obb;
        }

        for (int i = 0; i < numVertices; i++)
        {
            center += vertices[i];
        }

        center /= float(numVertices);

        for (int i = 0; i < numVertices; i++)
        {
            const Vec3 r = vertices[i] - center;
            A += outerProduct(r, r);
        }
    }

    Mat33 eigenVecs;
    Vec3 eigenVals;

    headerEigenDecomposition(A, eigenVecs, eigenVals, true);

    eigenVecs.column0.normalize();
    eigenVecs.column1.normalize();
    eigenVecs.column2.normalize();

    Mat33 inv = eigenVecs.getInverse();
    Bounds3 bounds;
    bounds.setEmpty();

    for (int i = 0; i < numVertices; i++)
    {
        Vec3 q = inv * vertices[i];
        bounds.include(q);
    }

    // break the symmetry

    Vec3 centerOffset = center - eigenVecs * bounds.getCenter();
    float eps = 0.0001f * bounds.getDimensions().length();

    int s = 1;

    float off0 = eigenVecs.column0.dot(centerOffset);
    float off1 = eigenVecs.column1.dot(centerOffset);
    float off2 = eigenVecs.column2.dot(centerOffset);

    if (off0 > eps)
    {
        eigenVecs.column0 = -eigenVecs.column0;
        s = -s;
    }
    else if (off0 > -eps)
    {
        s = 0; // symmetric
    }
    if (off1 > eps)
    {
        eigenVecs.column1 = -eigenVecs.column1;
        s = -s;
    }
    else if (off1 > -eps)
    {
        s = 0;
    }
    if (off2 > eps)
    {
        eigenVecs.column2 = -eigenVecs.column2;
        s = -s;
    }
    else if (off2 > -eps)
    {
        s = 0;
    }
    eigenVecs.column2 = eigenVecs.column0.cross(eigenVecs.column1); // force right handed frame

    if (sign)
    {
        *sign = s;
    }

    inv = eigenVecs.getInverse();
    bounds.setEmpty();

    for (int i = 0; i < numVertices; i++)
    {
        Vec3 q = inv * vertices[i];
        bounds.include(q);
    }
    obb.halfExtents = bounds.getHalfExtents();
    obb.trans.q = Quat(eigenVecs);
    obb.trans.p = obb.trans.q.rotate(bounds.getCenter());

    return obb;
}


struct CompareVec3X
{
    CUDA_CALLABLE bool operator()(const Vec3& a, const Vec3& b) const
    {
        return a.x < b.x;
    }
};

struct CompareVec3Y
{
    CUDA_CALLABLE bool operator()(const Vec3& a, const Vec3& b) const
    {
        return a.y < b.y;
    }
};

struct CompareVec3Z
{
    CUDA_CALLABLE bool operator()(const Vec3& a, const Vec3& b) const
    {
        return a.z < b.z;
    }
};

CUDA_CALLABLE inline bool header_rayTriangleIntersection2d(const Vec2& orig,
                                                           const Vec2& dir,
                                                           const Vec2& p0,
                                                           const Vec2& p1,
                                                           const Vec2& p2,
                                                           float& tMin,
                                                           float& tMax,
                                                           int& minEdgeNr,
                                                           float& uMin,
                                                           int& maxEdgeNr,
                                                           float& uMax)
{
    tMin = MaxFloat;
    tMax = -MaxFloat;
    minEdgeNr = -1;
    maxEdgeNr = -1;
    uMin = 0.0f;
    uMax = 0.0f;

    for (int i = 0; i < 3; i++)
    {
        Vec2 q0 = i == 0 ? p0 : (i == 1 ? p1 : p2);
        Vec2 q1 = i == 0 ? p1 : (i == 1 ? p2 : p0);

        // compute ray - (q0, q1) intersection

        Vec2 v0 = orig - q0;
        Vec2 v1 = q1 - q0;
        Vec2 v2(-dir.y, dir.x);

        float dot = v1.dot(v2);

        if (dot == 0.0f)
        {
            continue;
        }

        float t = v1.cross(v0) / dot;
        float u = v0.dot(v2) / dot;

        if (u >= 0.0f && u <= 1.0)
        {
            if (t < tMin)
            {
                tMin = t;
                minEdgeNr = i;
                uMin = u;
            }
            if (t > tMax)
            {
                tMax = t;
                maxEdgeNr = i;
                uMax = u;
            }
        }
    }

    return tMin <= tMax;
}


CUDA_CALLABLE inline bool header_rayTriangleIntersection2d(const Ray& ray,
                                                           const Vec3& p0,
                                                           const Vec3& p1,
                                                           const Vec3& p2,
                                                           float& tMin,
                                                           float& tMax,
                                                           int& minEdgeNr,
                                                           float& uMin,
                                                           int& maxEdgeNr,
                                                           float& uMax,
                                                           int projDim = 2)
{
    Vec2 q0(p0[(projDim + 1) % 3], p0[(projDim + 2) % 3]);
    Vec2 q1(p1[(projDim + 1) % 3], p1[(projDim + 2) % 3]);
    Vec2 q2(p2[(projDim + 1) % 3], p2[(projDim + 2) % 3]);

    Vec2 orig(ray.orig[(projDim + 1) % 3], ray.orig[(projDim + 2) % 3]);
    Vec2 dir(ray.dir[(projDim + 1) % 3], ray.dir[(projDim + 2) % 3]);

    return header_rayTriangleIntersection2d(orig, dir, q0, q1, q2, tMin, tMax, minEdgeNr, uMin, maxEdgeNr, uMax);
}


CUDA_CALLABLE inline bool header_rayRayIntersection2d(const Ray2& ray0, const Ray2& ray1, float& t0, float& t1)
{
    Vec2 v0 = ray0.orig - ray1.orig;
    Vec2 v1 = ray1.dir;
    Vec2 v2(-ray0.dir.y, ray0.dir.x);

    float dot = v1.dot(v2);

    if (dot == 0.0f)
    {
        return false;
    }

    t0 = v1.cross(v0) / dot;
    t1 = v0.dot(v2) / dot;

    return true;
}


CUDA_CALLABLE inline bool header_rayRayIntersection2d(
    const Ray& ray0, const Ray& ray1, const Vec3& projDir, float& t0, float& t1)
{
    Vec3 a0 = projDir.perp();
    Vec3 a1 = projDir.cross(a0).normalized();

    Vec2 orig0(ray0.orig.dot(a0), ray0.orig.dot(a1));
    Vec2 orig1(ray1.orig.dot(a0), ray1.orig.dot(a1));
    Vec2 dir0(ray0.dir.dot(a0), ray0.dir.dot(a1));
    Vec2 dir1(ray1.dir.dot(a0), ray1.dir.dot(a1));

    return header_rayRayIntersection2d(Ray2(orig0, dir0), Ray2(orig1, dir1), t0, t1);
}

CUDA_CALLABLE inline bool header_getSphereCenter(
    const Vec3& p0, const Vec3& p1, const Vec3& p2, float radius, bool above, Vec3& center)
{
    Vec3 n = (p1 - p0).cross(p2 - p0).normalized();
    if (n.isZero())
    {
        return false;
    }

    if (!above)
    {
        n = -n;
    }

    Vec3 circleCenter = header_getCircumCenterOfTriangle(p0, p1, p2);
    float circumRadius = (p0 - circleCenter).length();

    if (circumRadius > radius)
    {
        return false;
    }
    float h = sqrtf(radius * radius - circumRadius * circumRadius);
    center = circleCenter + n * h;

    return true;
}


}
