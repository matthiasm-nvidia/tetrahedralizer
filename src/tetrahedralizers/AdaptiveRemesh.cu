#include "TetDeviceData.h"

#include "utils/Geometry.h"

#include <cmath>

namespace tetrahedralizer
{
namespace
{

constexpr int kAdaptiveRemeshPasses = 16;
// Pre-split edge smoothing (commented out in runAdaptiveRemesh).
// constexpr int kAdaptiveSmoothIterations = 3;
// constexpr float kDefaultEdgeContraction = 0.2f;

__global__ void sampleNodeSizes(TetDeviceData data, const Vec3* normals, float maxSize)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)

    float size = maxSize;
    if (normals[nodeIndex].magnitudeSquared() > 0.0f)
    {
        Vec3 query = data.nodes[nodeIndex];
        Vec3 bary;
        Vec3 closestPos;
        int closestTri = -1;
        bool inside = false;
        if (header_queryClosestPoint(data.triangleBvh, query, 0.0f, data.meshVertices.buffer,
                                     reinterpret_cast<int*>(data.meshIndices.buffer), closestTri, bary, closestPos,
                                     inside) &&
            closestTri >= 0)
        {
            const std::uint32_t i0 = data.meshIndices[static_cast<std::size_t>(3 * closestTri + 0)];
            const std::uint32_t i1 = data.meshIndices[static_cast<std::size_t>(3 * closestTri + 1)];
            const std::uint32_t i2 = data.meshIndices[static_cast<std::size_t>(3 * closestTri + 2)];
            if (i0 < data.meshVertexSizes.size && i1 < data.meshVertexSizes.size && i2 < data.meshVertexSizes.size)
            {
                const float sampled = fminf(data.meshVertexSizes[i0],
                                            fminf(data.meshVertexSizes[i1], data.meshVertexSizes[i2]));
                if (sampled > 0.0f && isfinite(sampled))
                    size = sampled;
            }
        }
    }

    data.nodeSizes[nodeIndex] = size;
}

void sampleSurfaceNodeSizes(TetDeviceData& data, const Vec3* normals, float maxSize)
{
    data.nodeSizes.resize(static_cast<std::size_t>(data.numNodes), false);
    CUDA_LAUNCH(sampleNodeSizes, data.numNodes, data, normals, maxSize);
}

} // namespace

void runAdaptiveRemesh(TetDeviceData& data, const TetrahedralizerParams& params, float maxSize)
{
    if (data.numTets <= 0 || data.numNodes <= 0 || data.meshVertexSizes.size == 0)
        return;
    if (!computeNeighbors(data))
        throw std::runtime_error("Tet mesh has non-manifold faces before adaptive remesh");

    DeviceBuffer<Vec3> normals;
    // Closest-point sampling uses node positions, not smoothed normals. Smoothing here
    // moved the voxel mesh off-grid before measuring edge lengths.
    // const float contraction = params.edgeContraction > 0.0f ? params.edgeContraction : kDefaultEdgeContraction;
    // const Vec3* classifyNormals = params.useNormals ? nullptr : normals.buffer;
    // const Vec3* applyNormals = params.useNormals ? normals.buffer : nullptr;
    // computeSurfaceNormals(data, normals);
    // smoothEdges(data, kAdaptiveSmoothIterations, contraction, classifyNormals, applyNormals);
    (void)params;

    for (int pass = 0; pass < kAdaptiveRemeshPasses; ++pass)
    {
        computeSurfaceNormals(data, normals);
        sampleSurfaceNodeSizes(data, normals.buffer, maxSize);

        if (subdivideLongEdges(data, 0.0f, data.nodeSizes.buffer) == 0)
            break;

        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after adaptive split");
        separateBoundaryFaces(data);
    }

    normals.free();
}

} // namespace tetrahedralizer
