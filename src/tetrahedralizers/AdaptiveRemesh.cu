#include "TetDeviceData.h"

#include "utils/Geometry.h"

#include <cmath>

namespace tetrahedralizer
{
namespace
{

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

void runAdaptiveSplit(TetDeviceData& data, float maxSize, bool removeEmpty)
{
    if (data.numTets <= 0 || data.numNodes <= 0 || data.meshVertexSizes.size == 0)
        return;

    DeviceBuffer<Vec3> normals;
    computeSurfaceNormals(data, normals);
    sampleSurfaceNodeSizes(data, normals.buffer, maxSize);

    if (subdivideLongEdges(data, 0.0f, data.nodeSizes.buffer) > 0)
    {
        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after adaptive split");
        if (removeEmpty)
        {
            removeEmptyExteriorTets(data);
            if (!computeNeighbors(data))
                throw std::runtime_error("Tet mesh has non-manifold faces after adaptive empty-tet removal");
        }
        separateBoundaryFaces(data);
    }

    normals.free();
}

} // namespace tetrahedralizer
