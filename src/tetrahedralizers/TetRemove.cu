#include "TetDeviceData.h"

#include "utils/Geometry.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace tetrahedralizer
{
namespace
{

__device__ bool tetIntersectsInputMesh(const TetDeviceData& data, const Vec3& t0, const Vec3& t1, const Vec3& t2,
                                       const Vec3& t3)
{
    if (!data.triangleBvh.mRootNodes)
        return false;

    Bounds3 bounds(Empty);
    bounds.include(t0);
    bounds.include(t1);
    bounds.include(t2);
    bounds.include(t3);

    int stack[64];
    stack[0] = data.triangleBvh.mRootNodes[0];
    int count = 1;

    while (count > 0)
    {
        const int nodeIndex = stack[--count];
        const PackedNodeHalf lower = data.triangleBvh.mNodeLowers[nodeIndex];
        const PackedNodeHalf upper = data.triangleBvh.mNodeUppers[nodeIndex];
        if (upper.x < bounds.minimum.x || lower.x > bounds.maximum.x || upper.y < bounds.minimum.y ||
            lower.y > bounds.maximum.y || upper.z < bounds.minimum.z || lower.z > bounds.maximum.z)
            continue;

        if (lower.b)
        {
            const int triangleIndex = static_cast<int>(lower.i);
            const std::uint32_t i0 = data.meshIndices.buffer[3 * triangleIndex + 0];
            const std::uint32_t i1 = data.meshIndices.buffer[3 * triangleIndex + 1];
            const std::uint32_t i2 = data.meshIndices.buffer[3 * triangleIndex + 2];
            if (i0 < data.meshVertices.size && i1 < data.meshVertices.size && i2 < data.meshVertices.size &&
                header_triangleTetrahedronIntersection(data.meshVertices.buffer[i0], data.meshVertices.buffer[i1],
                                                       data.meshVertices.buffer[i2], t0, t1, t2, t3))
                return true;
        }
        else if (count <= 62)
        {
            stack[count++] = static_cast<int>(lower.i);
            stack[count++] = static_cast<int>(upper.i);
        }
    }
    return false;
}

__global__ void markTetsWithGeometry(TetDeviceData data, int* hasGeometry)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    const Vec3 t0 = data.nodes[data.tetIndices[4 * tetIndex + 0]];
    const Vec3 t1 = data.nodes[data.tetIndices[4 * tetIndex + 1]];
    const Vec3 t2 = data.nodes[data.tetIndices[4 * tetIndex + 2]];
    const Vec3 t3 = data.nodes[data.tetIndices[4 * tetIndex + 3]];
    hasGeometry[tetIndex] = tetIntersectsInputMesh(data, t0, t1, t2, t3) ? 1 : 0;
}

__global__ void seedKeptTets(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    data.firstNewTet[tetIndex] = data.tetInterior[tetIndex];
}

__global__ void floodKeptTets(TetDeviceData data, const int* hasGeometry)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    if (data.firstNewTet[tetIndex] == 0)
        return;
    for (int face = 0; face < 4; ++face)
    {
        const int neighbor = data.tetNeighbors[4 * tetIndex + face];
        if (neighbor < 0 || hasGeometry[neighbor] != 0)
            continue;
        if (atomicMax(&data.firstNewTet[neighbor], 1) == 0)
            data.anyChanged[0] = 1;
    }
}

__global__ void keepGeometryTets(TetDeviceData data, const int* hasGeometry)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    if (hasGeometry[tetIndex] != 0)
        data.firstNewTet[tetIndex] = 1;
}

__global__ void markUsedNodes(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    if (data.firstNewTet[tetIndex] == 0)
        return;
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 0], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 1], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 2], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 3], 1);
}

__global__ void compressNodes(TetDeviceData data, Vec3* compressedNodes)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    const int mappedId = data.firstSteiner[nodeIndex];
    if (mappedId == data.firstSteiner[nodeIndex + 1])
        return;
    compressedNodes[mappedId] = data.nodes[nodeIndex];
}

__global__ void compressTetIds(TetDeviceData data, int* compressedTetIds)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    const int mappedId = data.firstNewTet[tetIndex];
    if (mappedId == data.firstNewTet[tetIndex + 1])
        return;
    compressedTetIds[4 * mappedId + 0] = data.firstSteiner[data.tetIndices[4 * tetIndex + 0]];
    compressedTetIds[4 * mappedId + 1] = data.firstSteiner[data.tetIndices[4 * tetIndex + 1]];
    compressedTetIds[4 * mappedId + 2] = data.firstSteiner[data.tetIndices[4 * tetIndex + 2]];
    compressedTetIds[4 * mappedId + 3] = data.firstSteiner[data.tetIndices[4 * tetIndex + 3]];
}

__global__ void compressTetInterior(TetDeviceData data, int* compressedInterior)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    const int mappedId = data.firstNewTet[tetIndex];
    if (mappedId == data.firstNewTet[tetIndex + 1])
        return;
    compressedInterior[mappedId] = data.tetInterior[tetIndex];
}

} // namespace

void compactKeptTets(TetDeviceData& data)
{
    if (data.numTets <= 0)
    {
        data.numNodes = 0;
        data.nodes.free();
        data.tetIndices.free();
        data.tetInterior.free();
        data.tetNeighbors.free();
        return;
    }

    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), true);
    data.firstSteiner.resize(static_cast<std::size_t>(data.numNodes + 1), false);
    data.firstSteiner.setZero();
    CUDA_LAUNCH(markUsedNodes, data.numTets, data);

    thrust::device_ptr<int> tetMap(data.firstNewTet.buffer);
    thrust::device_ptr<int> nodeMap(data.firstSteiner.buffer);
    thrust::exclusive_scan(tetMap, tetMap + data.numTets + 1, tetMap);
    thrust::exclusive_scan(nodeMap, nodeMap + data.numNodes + 1, nodeMap);

    const int numNewTets = readDeviceInt(data.firstNewTet, data.numTets);
    const int numNewNodes = readDeviceInt(data.firstSteiner, data.numNodes);
    const bool keepInterior = data.tetInterior.size == static_cast<std::size_t>(data.numTets);

    DeviceBuffer<Vec3> compressedNodes;
    DeviceBuffer<int> compressedTetIds;
    DeviceBuffer<int> compressedInterior;
    compressedNodes.resize(static_cast<std::size_t>(numNewNodes), false);
    compressedTetIds.resize(static_cast<std::size_t>(numNewTets) * 4, false);
    CUDA_LAUNCH(compressNodes, data.numNodes, data, compressedNodes.buffer);
    CUDA_LAUNCH(compressTetIds, data.numTets, data, compressedTetIds.buffer);
    if (keepInterior)
    {
        compressedInterior.resize(static_cast<std::size_t>(numNewTets), false);
        CUDA_LAUNCH(compressTetInterior, data.numTets, data, compressedInterior.buffer);
    }

    data.nodes.swap(compressedNodes);
    data.tetIndices.swap(compressedTetIds);
    if (keepInterior)
        data.tetInterior.swap(compressedInterior);
    else
        data.tetInterior.free();
    data.numNodes = numNewNodes;
    data.numTets = numNewTets;
    compressedNodes.free();
    compressedTetIds.free();
    compressedInterior.free();
}

void removeEmptyExteriorTets(TetDeviceData& data)
{
    if (data.numTets <= 0 || data.numNodes <= 0)
        return;
    if (data.tetInterior.size != static_cast<std::size_t>(data.numTets) || data.numTriangles <= 0 ||
        !data.triangleBvh.mRootNodes)
        return;
    if (data.tetNeighbors.size != static_cast<std::size_t>(data.numTets) * 4)
        return;

    DeviceBuffer<int> hasGeometry;
    hasGeometry.resize(static_cast<std::size_t>(data.numTets), false);
    CUDA_LAUNCH(markTetsWithGeometry, data.numTets, data, hasGeometry.buffer);

    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstNewTet.setZero();
    CUDA_LAUNCH(seedKeptTets, data.numTets, data);

    data.anyChanged.resize(1, false);
    for (int iter = 0; iter < data.numTets; ++iter)
    {
        data.anyChanged.setZero();
        CUDA_LAUNCH(floodKeptTets, data.numTets, data, hasGeometry.buffer);
        if (readDeviceInt(data.anyChanged, 0) == 0)
            break;
    }

    CUDA_LAUNCH(keepGeometryTets, data.numTets, data, hasGeometry.buffer);
    hasGeometry.free();
    compactKeptTets(data);
}

} // namespace tetrahedralizer
