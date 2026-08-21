#include "tetrahedralizer/SizeField.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tetrahedralizer
{
namespace
{

constexpr float kMinEdgeLength = 1.0e-12f;
constexpr float kMinAreaSquared = 1.0e-24f;

int triangleCount(std::size_t indexCount)
{
    return static_cast<int>(indexCount / 3);
}

bool validCorner(int index, int vertexCount)
{
    return index >= 0 && index < vertexCount;
}

float resolveMaxSize(const std::vector<Vec3>& positions, float maxSize)
{
    if (maxSize > 0.0f && std::isfinite(maxSize))
        return maxSize;

    Bounds3 bounds(Empty);
    for (const Vec3& point : positions)
        bounds.include(point);
    const float diagonal = bounds.getDimensions().length();
    return diagonal > 0.0f && std::isfinite(diagonal) ? diagonal : 1.0f;
}

void accumulateVertexNormals(const std::vector<Vec3>& positions,
                             const std::vector<std::uint32_t>& triangle_indices,
                             std::vector<Vec3>& normals)
{
    const int vertexCount = static_cast<int>(positions.size());
    const int numTriangles = triangleCount(triangle_indices.size());
    normals.assign(static_cast<std::size_t>(vertexCount), Vec3(Zero));

    for (int triangle = 0; triangle < numTriangles; ++triangle)
    {
        const int i0 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 0)]);
        const int i1 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 1)]);
        const int i2 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 2)]);
        if (!validCorner(i0, vertexCount) || !validCorner(i1, vertexCount) || !validCorner(i2, vertexCount))
            continue;

        const Vec3& p0 = positions[static_cast<std::size_t>(i0)];
        const Vec3 faceNormal = (positions[static_cast<std::size_t>(i1)] - p0)
                                    .cross(positions[static_cast<std::size_t>(i2)] - p0);
        if (faceNormal.magnitudeSquared() < kMinAreaSquared)
            continue;

        normals[static_cast<std::size_t>(i0)] += faceNormal;
        normals[static_cast<std::size_t>(i1)] += faceNormal;
        normals[static_cast<std::size_t>(i2)] += faceNormal;
    }

    for (Vec3& normal : normals)
    {
        if (normal.magnitudeSquared() > 0.0f)
            normal.normalize();
    }
}

void accumulateMaxTurning(const std::vector<Vec3>& positions,
                          const std::vector<std::uint32_t>& triangle_indices,
                          const std::vector<Vec3>& normals,
                          std::vector<float>& maxKappa)
{
    const int vertexCount = static_cast<int>(positions.size());
    const int numTriangles = triangleCount(triangle_indices.size());
    maxKappa.assign(static_cast<std::size_t>(vertexCount), 0.0f);

    auto considerEdge = [&](int i0, int i1) {
        const Vec3& n0 = normals[static_cast<std::size_t>(i0)];
        const Vec3& n1 = normals[static_cast<std::size_t>(i1)];
        if (n0.magnitudeSquared() == 0.0f || n1.magnitudeSquared() == 0.0f)
            return;

        const float length = (positions[static_cast<std::size_t>(i1)] - positions[static_cast<std::size_t>(i0)]).length();
        if (!(length > kMinEdgeLength))
            return;

        // |n0 - n1| = 2 sin(α/2) for unit normals, so κ = 2 sin(α/2) / L.
        const float kappa = (n0 - n1).length() / length;
        if (!std::isfinite(kappa))
            return;

        float& k0 = maxKappa[static_cast<std::size_t>(i0)];
        float& k1 = maxKappa[static_cast<std::size_t>(i1)];
        if (kappa > k0)
            k0 = kappa;
        if (kappa > k1)
            k1 = kappa;
    };

    for (int triangle = 0; triangle < numTriangles; ++triangle)
    {
        const int i0 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 0)]);
        const int i1 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 1)]);
        const int i2 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 2)]);
        if (!validCorner(i0, vertexCount) || !validCorner(i1, vertexCount) || !validCorner(i2, vertexCount))
            continue;

        considerEdge(i0, i1);
        considerEdge(i1, i2);
        considerEdge(i2, i0);
    }
}

void kappaToSize(std::vector<float>& sizes, const std::vector<float>& maxKappa, float geometricError, float minSize,
                 float maxSize)
{
    const float eightError = 8.0f * geometricError;
    for (std::size_t i = 0; i < sizes.size(); ++i)
    {
        const float kappa = maxKappa[i];
        float size = maxSize;
        if (geometricError > 0.0f && kappa > 0.0f)
            size = std::sqrt(eightError / kappa);
        if (!std::isfinite(size) || size > maxSize)
            size = maxSize;
        if (minSize > 0.0f && size < minSize)
            size = minSize;
        sizes[i] = size;
    }
}

void smoothSizes(std::vector<float>& sizes, const std::vector<std::uint32_t>& triangle_indices, int iterations)
{
    if (iterations <= 0 || sizes.empty())
        return;

    const int vertexCount = static_cast<int>(sizes.size());
    const int numTriangles = triangleCount(triangle_indices.size());
    std::vector<float> sums(sizes.size());
    std::vector<int> counts(sizes.size());

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (int triangle = 0; triangle < numTriangles; ++triangle)
        {
            const int i0 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 0)]);
            const int i1 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 1)]);
            const int i2 = static_cast<int>(triangle_indices[static_cast<std::size_t>(3 * triangle + 2)]);
            if (!validCorner(i0, vertexCount) || !validCorner(i1, vertexCount) || !validCorner(i2, vertexCount))
                continue;

            const float mean =
                (sizes[static_cast<std::size_t>(i0)] + sizes[static_cast<std::size_t>(i1)] +
                 sizes[static_cast<std::size_t>(i2)]) *
                (1.0f / 3.0f);
            sums[static_cast<std::size_t>(i0)] += mean;
            sums[static_cast<std::size_t>(i1)] += mean;
            sums[static_cast<std::size_t>(i2)] += mean;
            ++counts[static_cast<std::size_t>(i0)];
            ++counts[static_cast<std::size_t>(i1)];
            ++counts[static_cast<std::size_t>(i2)];
        }

        for (int vertex = 0; vertex < vertexCount; ++vertex)
        {
            if (counts[static_cast<std::size_t>(vertex)] > 0)
                sizes[static_cast<std::size_t>(vertex)] =
                    sums[static_cast<std::size_t>(vertex)] /
                    static_cast<float>(counts[static_cast<std::size_t>(vertex)]);
        }
    }
}

} // namespace

std::vector<float> computeSurfaceSizeField(const std::vector<Vec3>& positions,
                                           const std::vector<std::uint32_t>& triangle_indices,
                                           const SizeFieldParams& params)
{
    std::vector<float> sizes(positions.size(), 0.0f);
    if (positions.empty() || triangle_indices.size() < 3)
        return sizes;

    const float maxSize = resolveMaxSize(positions, params.maxSize);
    const float minSize = params.minSize > 0.0f && std::isfinite(params.minSize) ? params.minSize : 0.0f;
    const float geometricError =
        params.geometricError > 0.0f && std::isfinite(params.geometricError) ? params.geometricError : 0.0f;

    std::vector<Vec3> normals;
    accumulateVertexNormals(positions, triangle_indices, normals);

    std::vector<float> maxKappa;
    accumulateMaxTurning(positions, triangle_indices, normals, maxKappa);

    kappaToSize(sizes, maxKappa, geometricError, minSize, maxSize);
    smoothSizes(sizes, triangle_indices, params.smoothingIterations);
    return sizes;
}

} // namespace tetrahedralizer
