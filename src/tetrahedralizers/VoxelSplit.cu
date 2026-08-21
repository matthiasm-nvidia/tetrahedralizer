#include "TetDeviceData.h"

#include "utils/Geometry.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace tetrahedralizer
{
namespace
{

__device__ __constant__ int kVoxelCorners[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};

// Face order: -X, +X, -Y, +Y, -Z, +Z.
__device__ __constant__ int kVoxelFaceOffsets[6][3] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
};

// Local corner pairs occupying the same grid point across each voxel face.
__device__ __constant__ int kVoxelFaceCornerPairs[6][4][2] = {
    {{0, 1}, {3, 2}, {4, 5}, {7, 6}},
    {{1, 0}, {2, 3}, {5, 4}, {6, 7}},
    {{0, 3}, {1, 2}, {4, 7}, {5, 6}},
    {{3, 0}, {2, 1}, {7, 4}, {6, 5}},
    {{0, 4}, {1, 5}, {2, 6}, {3, 7}},
    {{4, 0}, {5, 1}, {6, 2}, {7, 3}},
};

// The two mirror-image five-tet cube decompositions. Neighboring voxels use
// opposite patterns so their shared face diagonals agree.
__device__ __constant__ int kTetCorners[2][5][4] = {
    {
        {0, 1, 3, 4},
        {1, 2, 3, 6},
        {1, 4, 5, 6},
        {3, 4, 6, 7},
        {1, 3, 4, 6},
    },
    {
        {1, 0, 2, 5},
        {0, 3, 2, 7},
        {0, 5, 4, 7},
        {2, 5, 7, 6},
        {0, 2, 5, 7},
    },
};

__device__ bool faceIntersectsInputMesh(const TetDeviceData& data, const Vec3& center, const Vec3& extents)
{
    int stack[64];
    stack[0] = data.triangleBvh.mRootNodes[0];
    int count = 1;

    while (count > 0)
    {
        const int nodeIndex = stack[--count];
        const PackedNodeHalf lower = data.triangleBvh.mNodeLowers[nodeIndex];
        const PackedNodeHalf upper = data.triangleBvh.mNodeUppers[nodeIndex];
        if (upper.x < center.x - extents.x || lower.x > center.x + extents.x ||
            upper.y < center.y - extents.y || lower.y > center.y + extents.y ||
            upper.z < center.z - extents.z || lower.z > center.z + extents.z)
            continue;

        if (lower.b)
        {
            const int triangleIndex = static_cast<int>(lower.i);
            const std::uint32_t i0 = data.meshIndices.buffer[3 * triangleIndex + 0];
            const std::uint32_t i1 = data.meshIndices.buffer[3 * triangleIndex + 1];
            const std::uint32_t i2 = data.meshIndices.buffer[3 * triangleIndex + 2];
            if (i0 < data.meshVertices.size && i1 < data.meshVertices.size && i2 < data.meshVertices.size &&
                header_boxTriangleIntersection(data.meshVertices.buffer[i0], data.meshVertices.buffer[i1],
                                               data.meshVertices.buffer[i2], center, extents))
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

__global__ void findConnectedVoxelFaces(TetDeviceData data)
{
    CUDA_THREAD_GUARD(index, data.numVoxels * 6)
    const int voxelIndex = index / 6;
    const int face = index % 6;

    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    const int nx = x + kVoxelFaceOffsets[face][0];
    const int ny = y + kVoxelFaceOffsets[face][1];
    const int nz = z + kVoxelFaceOffsets[face][2];
    if (findCoord(data.voxels, packCoords(nx, ny, nz)) < 0)
    {
        data.connectedVoxelFaces[index] = 0;
        return;
    }

    const bool internal = !getCell(data.geomCells.buffer, data, x, y, z);
    const bool neighborInternal = !getCell(data.geomCells.buffer, data, nx, ny, nz);
    if (internal || neighborInternal)
    {
        data.connectedVoxelFaces[index] = 1;
        return;
    }

    const float half = data.gridSpacing * 0.5f;
    const float margin = data.gridSpacing * kVoxelOverlapMargin;
    Vec3 center(data.worldOrigin.x + (static_cast<float>(x) + 0.5f) * data.gridSpacing,
                data.worldOrigin.y + (static_cast<float>(y) + 0.5f) * data.gridSpacing,
                data.worldOrigin.z + (static_cast<float>(z) + 0.5f) * data.gridSpacing);
    center.x += static_cast<float>(kVoxelFaceOffsets[face][0]) * half;
    center.y += static_cast<float>(kVoxelFaceOffsets[face][1]) * half;
    center.z += static_cast<float>(kVoxelFaceOffsets[face][2]) * half;

    Vec3 extents(half + margin, half + margin, half + margin);
    if (kVoxelFaceOffsets[face][0] != 0)
        extents.x = margin;
    else if (kVoxelFaceOffsets[face][1] != 0)
        extents.y = margin;
    else
        extents.z = margin;
    data.connectedVoxelFaces[index] = faceIntersectsInputMesh(data, center, extents) ? 1 : 0;
}

__global__ void initializeMergedCornerIds(TetDeviceData data)
{
    CUDA_THREAD_GUARD(cornerIndex, data.numVoxels * 8)
    data.mergedCornerIds[cornerIndex] = cornerIndex;
}

__global__ void mergeConnectedVoxelCorners(TetDeviceData data)
{
    CUDA_THREAD_GUARD(index, data.numVoxels * 6)
    if (data.connectedVoxelFaces[index] == 0)
        return;

    const int voxelIndex = index / 6;
    const int face = index % 6;
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    const int neighborIndex =
        findCoord(data.voxels, packCoords(x + kVoxelFaceOffsets[face][0], y + kVoxelFaceOffsets[face][1],
                                         z + kVoxelFaceOffsets[face][2]));
    if (neighborIndex < 0)
        return;

    for (int corner = 0; corner < 4; ++corner)
    {
        const int id0 = 8 * voxelIndex + kVoxelFaceCornerPairs[face][corner][0];
        const int id1 = 8 * neighborIndex + kVoxelFaceCornerPairs[face][corner][1];
        const int mergedId = Min(data.mergedCornerIds[id0], data.mergedCornerIds[id1]);
        if (atomicMin(&data.mergedCornerIds[id0], mergedId) > mergedId)
            data.anyChanged[0] = 1;
        if (atomicMin(&data.mergedCornerIds[id1], mergedId) > mergedId)
            data.anyChanged[0] = 1;
    }
}

__global__ void markMergedNodes(TetDeviceData data)
{
    CUDA_THREAD_GUARD(cornerIndex, data.numVoxels * 8)
    atomicMax(&data.nodeOffsets[data.mergedCornerIds[cornerIndex]], 1);
}

__global__ void createSplitNodes(TetDeviceData data)
{
    CUDA_THREAD_GUARD(cornerIndex, data.numVoxels * 8)
    const int mergedId = data.mergedCornerIds[cornerIndex];
    const int nodeId = data.nodeOffsets[mergedId];
    data.voxelCornerNodes[cornerIndex] = nodeId;
    if (mergedId != cornerIndex)
        return;

    const int voxelIndex = cornerIndex / 8;
    const int corner = cornerIndex % 8;
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    data.nodes[nodeId] =
        Vec3(data.worldOrigin.x + static_cast<float>(x + kVoxelCorners[corner][0]) * data.gridSpacing,
             data.worldOrigin.y + static_cast<float>(y + kVoxelCorners[corner][1]) * data.gridSpacing,
             data.worldOrigin.z + static_cast<float>(z + kVoxelCorners[corner][2]) * data.gridSpacing);
}

__global__ void createTets(TetDeviceData data)
{
    CUDA_THREAD_GUARD(voxelIndex, data.numVoxels)
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    const int pattern = (x + y + z) & 1;

    int cornerIds[8];
    for (int corner = 0; corner < 8; ++corner)
        cornerIds[corner] = data.voxelCornerNodes[8 * voxelIndex + corner];

    for (int tet = 0; tet < 5; ++tet)
    {
        int ids[4];
        for (int corner = 0; corner < 4; ++corner)
            ids[corner] = cornerIds[kTetCorners[pattern][tet][corner]];

        const Vec3 p0 = data.nodes[ids[0]];
        const Vec3 p1 = data.nodes[ids[1]];
        const Vec3 p2 = data.nodes[ids[2]];
        const Vec3 p3 = data.nodes[ids[3]];
        if ((p1 - p0).cross(p2 - p0).dot(p3 - p0) < 0.0f)
        {
            const int swap = ids[2];
            ids[2] = ids[3];
            ids[3] = swap;
        }

        const int outputIndex = (voxelIndex * 5 + tet) * 4;
        for (int corner = 0; corner < 4; ++corner)
            data.tetIndices[outputIndex + corner] = ids[corner];
        data.tetInterior[voxelIndex * 5 + tet] = getCell(data.geomCells.buffer, data, x, y, z) ? 0 : 1;
    }
}

__global__ void computeTriangleBounds(TetDeviceData data)
{
    CUDA_THREAD_GUARD(triangleIndex, data.numTriangles)
    const std::uint32_t i0 = data.meshIndices[3 * triangleIndex + 0];
    const std::uint32_t i1 = data.meshIndices[3 * triangleIndex + 1];
    const std::uint32_t i2 = data.meshIndices[3 * triangleIndex + 2];

    if (i0 >= data.meshVertices.size || i1 >= data.meshVertices.size || i2 >= data.meshVertices.size)
    {
        data.triangleBoundsLowers[triangleIndex] = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        data.triangleBoundsUppers[triangleIndex] = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const Vec3 p0 = data.meshVertices[i0];
    const Vec3 p1 = data.meshVertices[i1];
    const Vec3 p2 = data.meshVertices[i2];
    data.triangleBoundsLowers[triangleIndex] =
        Vec4(Min(p0.x, p1.x, p2.x), Min(p0.y, p1.y, p2.y), Min(p0.z, p1.z, p2.z), 0.0f);
    data.triangleBoundsUppers[triangleIndex] =
        Vec4(Max(p0.x, p1.x, p2.x), Max(p0.y, p1.y, p2.y), Max(p0.z, p1.z, p2.z), 0.0f);
}

} // namespace

void buildInputMeshBvh(TetDeviceData& data)
{
    data.triangleBoundsLowers.resize(static_cast<std::size_t>(data.numTriangles), false);
    data.triangleBoundsUppers.resize(static_cast<std::size_t>(data.numTriangles), false);
    CUDA_LAUNCH(computeTriangleBounds, data.numTriangles, data);

    BVHBuilderGPU bvhBuilder;
    bvhBuilder.build(data.triangleBvh, data.triangleBoundsLowers.buffer, data.triangleBoundsUppers.buffer,
                     data.numTriangles);
}

void createSplitVoxelNodes(TetDeviceData& data)
{
    const int cornerCount = data.numVoxels * 8;
    data.connectedVoxelFaces.resize(static_cast<std::size_t>(data.numVoxels) * 6, false);
    CUDA_LAUNCH(findConnectedVoxelFaces, data.numVoxels * 6, data);

    data.mergedCornerIds.resize(static_cast<std::size_t>(cornerCount), false);
    CUDA_LAUNCH(initializeMergedCornerIds, cornerCount, data);
    while (true)
    {
        data.anyChanged.setZero();
        CUDA_LAUNCH(mergeConnectedVoxelCorners, data.numVoxels * 6, data);
        if (readDeviceInt(data.anyChanged, 0) == 0)
            break;
    }

    data.nodeOffsets.resize(static_cast<std::size_t>(cornerCount + 1), false);
    data.nodeOffsets.setZero();
    CUDA_LAUNCH(markMergedNodes, cornerCount, data);
    thrust::device_ptr<int> nodeOffsets(data.nodeOffsets.buffer);
    thrust::exclusive_scan(nodeOffsets, nodeOffsets + cornerCount + 1, nodeOffsets);

    data.numNodes = readDeviceInt(data.nodeOffsets, cornerCount);
    data.nodes.resize(static_cast<std::size_t>(data.numNodes), false);
    data.voxelCornerNodes.resize(static_cast<std::size_t>(cornerCount), false);
    CUDA_LAUNCH(createSplitNodes, cornerCount, data);
}

void createFiveTets(TetDeviceData& data)
{
    if (data.numVoxels > std::numeric_limits<int>::max() / 5)
        throw std::runtime_error("Tet count exceeds the supported range");
    data.numTets = data.numVoxels * 5;
    data.tetIndices.resize(static_cast<std::size_t>(data.numTets) * 4, false);
    data.tetInterior.resize(static_cast<std::size_t>(data.numTets), false);
    CUDA_LAUNCH(createTets, data.numVoxels, data);
}

} // namespace tetrahedralizer
