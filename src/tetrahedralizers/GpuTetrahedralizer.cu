#include "GpuTetrahedralizer.h"

#include "tetrahedralizer/Tetrahedralizer.h"
#include "utils/CudaUtils.h"
#include "utils/Geometry.h"
#include "utils/GpuBVH.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <vector>

namespace tetrahedralizer
{
namespace
{

constexpr std::uint64_t kCoordMask = (1ull << 21) - 1ull;

__device__ __constant__ int kVoxelCorners[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
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

__device__ __constant__ int kTetEdges[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
};

struct DeviceVec3
{
    float x;
    float y;
    float z;

    __device__ DeviceVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __device__ DeviceVec3(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}
    __device__ DeviceVec3 operator-(const DeviceVec3& v) const { return DeviceVec3(x - v.x, y - v.y, z - v.z); }
    __device__ float dot(const DeviceVec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __device__ DeviceVec3 cross(const DeviceVec3& v) const
    {
        return DeviceVec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
};

__device__ float min3(float a, float b, float c)
{
    return fminf(a, fminf(b, c));
}

__device__ float max3(float a, float b, float c)
{
    return fmaxf(a, fmaxf(b, c));
}

__device__ float minMax(float a, float b, float c)
{
    return fminf(-max3(a, b, c), min3(a, b, c));
}

__device__ bool boxTriangleIntersection(const DeviceVec3& p0, const DeviceVec3& p1, const DeviceVec3& p2,
                                        const DeviceVec3& center, const DeviceVec3& extents)
{
    const DeviceVec3 v0 = p0 - center;
    const DeviceVec3 v1 = p1 - center;
    const DeviceVec3 v2 = p2 - center;
    const DeviceVec3 edges[3] = {p1 - p0, p2 - p1, p0 - p2};

    const DeviceVec3 normal = edges[0].cross(edges[1]);
    const float planeRadius =
        extents.x * fabsf(normal.x) + extents.y * fabsf(normal.y) + extents.z * fabsf(normal.z);
    const float planeDistance = normal.dot(v0);
    if (planeDistance > planeRadius || planeDistance < -planeRadius)
        return false;

    if (max3(v0.x, v1.x, v2.x) < -extents.x || min3(v0.x, v1.x, v2.x) > extents.x ||
        max3(v0.y, v1.y, v2.y) < -extents.y || min3(v0.y, v1.y, v2.y) > extents.y ||
        max3(v0.z, v1.z, v2.z) < -extents.z || min3(v0.z, v1.z, v2.z) > extents.z)
        return false;

    for (int edge = 0; edge < 3; ++edge)
    {
        const DeviceVec3 f = edges[edge];
        DeviceVec3 axis(0.0f, -f.z, f.y);
        float radius = extents.y * fabsf(f.z) + extents.z * fabsf(f.y);
        if (minMax(v0.dot(axis), v1.dot(axis), v2.dot(axis)) > radius)
            return false;

        axis = DeviceVec3(f.z, 0.0f, -f.x);
        radius = extents.x * fabsf(f.z) + extents.z * fabsf(f.x);
        if (minMax(v0.dot(axis), v1.dot(axis), v2.dot(axis)) > radius)
            return false;

        axis = DeviceVec3(-f.y, f.x, 0.0f);
        radius = extents.x * fabsf(f.y) + extents.y * fabsf(f.x);
        if (minMax(v0.dot(axis), v1.dot(axis), v2.dot(axis)) > radius)
            return false;
    }

    return true;
}

__host__ __device__ std::uint64_t packCoords(int x, int y, int z)
{
    return (static_cast<std::uint64_t>(x) << 42) |
           (static_cast<std::uint64_t>(y) << 21) |
           static_cast<std::uint64_t>(z);
}

__host__ __device__ void unpackCoords(std::uint64_t packed, int& x, int& y, int& z)
{
    x = static_cast<int>(packed >> 42);
    y = static_cast<int>((packed >> 21) & kCoordMask);
    z = static_cast<int>(packed & kCoordMask);
}

struct TetDeviceData
{
    int numTriangles = 0;
    int numVoxels = 0;
    int numNodes = 0;
    int numTets = 0;
    int numEdges = 0;
    int gridNumX = 0;
    int gridNumY = 0;
    int gridNumZ = 0;
    int gridNumCells = 0;
    float gridSpacing = 0.0f;
    float invGridSpacing = 0.0f;
    Vec3 worldOrigin;

    DeviceBuffer<Vec3> meshVertices;
    DeviceBuffer<std::uint32_t> meshIndices;
    DeviceBuffer<std::uint64_t> voxels;
    DeviceBuffer<int> firstVoxel;
    DeviceBuffer<std::uint64_t> cornerCoords;
    DeviceBuffer<Vec3> nodes;
    DeviceBuffer<int> tetIndices;
    DeviceBuffer<std::uint64_t> edges;
    DeviceBuffer<int> firstCutVertex;
    DeviceBuffer<int> edgeCutVertices;
    DeviceBuffer<Vec4> triangleBoundsLowers;
    DeviceBuffer<Vec4> triangleBoundsUppers;
    GpuBVH triangleBvh;
    DeviceBuffer<std::uint32_t> geomCells;
    DeviceBuffer<std::uint32_t> airCells;
    DeviceBuffer<std::uint32_t> morphCells;
    DeviceBuffer<int> anyChanged;
    DeviceBuffer<int> voxelCounter;

    void free()
    {
        meshVertices.free();
        meshIndices.free();
        voxels.free();
        firstVoxel.free();
        cornerCoords.free();
        nodes.free();
        tetIndices.free();
        edges.free();
        firstCutVertex.free();
        edgeCutVertices.free();
        triangleBoundsLowers.free();
        triangleBoundsUppers.free();
        triangleBvh.free();
        geomCells.free();
        airCells.free();
        morphCells.free();
        anyChanged.free();
        voxelCounter.free();
        numTriangles = 0;
        numVoxels = 0;
        numNodes = 0;
        numTets = 0;
        numEdges = 0;
        gridNumX = 0;
        gridNumY = 0;
        gridNumZ = 0;
        gridNumCells = 0;
    }
};

__device__ int findCoord(const DeviceBuffer<std::uint64_t>& coords, std::uint64_t target)
{
    int first = 0;
    int last = static_cast<int>(coords.size);
    while (first < last)
    {
        const int middle = first + (last - first) / 2;
        if (coords.buffer[middle] < target)
            first = middle + 1;
        else
            last = middle;
    }
    return first < static_cast<int>(coords.size) && coords.buffer[first] == target ? first : -1;
}

__device__ int cellIndex(const TetDeviceData& data, int x, int y, int z)
{
    return (x * data.gridNumY + y) * data.gridNumZ + z;
}

__device__ bool getCell(const std::uint32_t* cells, const TetDeviceData& data, int x, int y, int z)
{
    if (x < 0 || x >= data.gridNumX || y < 0 || y >= data.gridNumY || z < 0 || z >= data.gridNumZ)
        return false;
    const int index = cellIndex(data, x, y, z);
    return (cells[index >> 5] & (1u << (index & 31))) != 0;
}

__device__ void setCell(std::uint32_t* cells, const TetDeviceData& data, int x, int y, int z)
{
    const int index = cellIndex(data, x, y, z);
    atomicOr(&cells[index >> 5], 1u << (index & 31));
}

__global__ void createTriangleVoxels(TetDeviceData data, bool countOnly)
{
    CUDA_THREAD_GUARD(triangleIndex, data.numTriangles)

    const std::uint32_t i0 = data.meshIndices[3 * triangleIndex + 0];
    const std::uint32_t i1 = data.meshIndices[3 * triangleIndex + 1];
    const std::uint32_t i2 = data.meshIndices[3 * triangleIndex + 2];
    if (i0 >= data.meshVertices.size || i1 >= data.meshVertices.size || i2 >= data.meshVertices.size)
    {
        if (countOnly)
            data.firstVoxel[triangleIndex] = 0;
        return;
    }

    const DeviceVec3 p0(data.meshVertices[i0]);
    const DeviceVec3 p1(data.meshVertices[i1]);
    const DeviceVec3 p2(data.meshVertices[i2]);
    const DeviceVec3 origin(data.worldOrigin);

    const int x0 = static_cast<int>(floorf((min3(p0.x, p1.x, p2.x) - origin.x) * data.invGridSpacing));
    const int y0 = static_cast<int>(floorf((min3(p0.y, p1.y, p2.y) - origin.y) * data.invGridSpacing));
    const int z0 = static_cast<int>(floorf((min3(p0.z, p1.z, p2.z) - origin.z) * data.invGridSpacing));
    const int x1 = static_cast<int>(floorf((max3(p0.x, p1.x, p2.x) - origin.x) * data.invGridSpacing));
    const int y1 = static_cast<int>(floorf((max3(p0.y, p1.y, p2.y) - origin.y) * data.invGridSpacing));
    const int z1 = static_cast<int>(floorf((max3(p0.z, p1.z, p2.z) - origin.z) * data.invGridSpacing));

    int count = 0;
    int outputIndex = data.numVoxels + data.firstVoxel[triangleIndex];
    const DeviceVec3 halfExtents(data.gridSpacing * 0.5f, data.gridSpacing * 0.5f, data.gridSpacing * 0.5f);
    for (int x = x0; x <= x1; ++x)
    {
        for (int y = y0; y <= y1; ++y)
        {
            for (int z = z0; z <= z1; ++z)
            {
                const DeviceVec3 center(
                    origin.x + (static_cast<float>(x) + 0.5f) * data.gridSpacing,
                    origin.y + (static_cast<float>(y) + 0.5f) * data.gridSpacing,
                    origin.z + (static_cast<float>(z) + 0.5f) * data.gridSpacing);
                if (!boxTriangleIntersection(p0, p1, p2, center, halfExtents))
                    continue;

                if (!countOnly)
                    data.voxels[outputIndex + count] = packCoords(x, y, z);
                ++count;
            }
        }
    }

    if (countOnly)
        data.firstVoxel[triangleIndex] = count;
}

__global__ void createCornerCoords(TetDeviceData data)
{
    CUDA_THREAD_GUARD(index, data.numVoxels * 8)
    const int voxelIndex = index / 8;
    const int corner = index % 8;
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    data.cornerCoords[index] =
        packCoords(x + kVoxelCorners[corner][0],
                   y + kVoxelCorners[corner][1],
                   z + kVoxelCorners[corner][2]);
}

__global__ void createNodes(TetDeviceData data)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    int x, y, z;
    unpackCoords(data.cornerCoords[nodeIndex], x, y, z);
    data.nodes[nodeIndex].x = data.worldOrigin.x + static_cast<float>(x) * data.gridSpacing;
    data.nodes[nodeIndex].y = data.worldOrigin.y + static_cast<float>(y) * data.gridSpacing;
    data.nodes[nodeIndex].z = data.worldOrigin.z + static_cast<float>(z) * data.gridSpacing;
}

__global__ void createTets(TetDeviceData data)
{
    CUDA_THREAD_GUARD(voxelIndex, data.numVoxels)
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    const int pattern = (x + y + z) & 1;

    int cornerIds[8];
    for (int corner = 0; corner < 8; ++corner)
    {
        cornerIds[corner] = findCoord(
            data.cornerCoords,
            packCoords(x + kVoxelCorners[corner][0],
                       y + kVoxelCorners[corner][1],
                       z + kVoxelCorners[corner][2]));
    }

    for (int tet = 0; tet < 5; ++tet)
    {
        int ids[4];
        for (int corner = 0; corner < 4; ++corner)
            ids[corner] = cornerIds[kTetCorners[pattern][tet][corner]];

        const DeviceVec3 p0(data.nodes[ids[0]]);
        const DeviceVec3 p1(data.nodes[ids[1]]);
        const DeviceVec3 p2(data.nodes[ids[2]]);
        const DeviceVec3 p3(data.nodes[ids[3]]);
        if ((p1 - p0).cross(p2 - p0).dot(p3 - p0) < 0.0f)
        {
            const int swap = ids[2];
            ids[2] = ids[3];
            ids[3] = swap;
        }

        const int outputIndex = (voxelIndex * 5 + tet) * 4;
        for (int corner = 0; corner < 4; ++corner)
            data.tetIndices[outputIndex + corner] = ids[corner];
    }
}

__host__ __device__ std::uint64_t packEdge(int id0, int id1)
{
    const std::uint32_t lower = static_cast<std::uint32_t>(id0 < id1 ? id0 : id1);
    const std::uint32_t upper = static_cast<std::uint32_t>(id0 < id1 ? id1 : id0);
    return (static_cast<std::uint64_t>(lower) << 32) | upper;
}

__global__ void createTetEdges(TetDeviceData data)
{
    CUDA_THREAD_GUARD(index, data.numTets * 6)
    const int tetIndex = index / 6;
    const int edgeIndex = index % 6;
    const int id0 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][0]];
    const int id1 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][1]];
    data.edges[index] = packEdge(id0, id1);
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
        Vec4(min3(p0.x, p1.x, p2.x), min3(p0.y, p1.y, p2.y), min3(p0.z, p1.z, p2.z), 0.0f);
    data.triangleBoundsUppers[triangleIndex] =
        Vec4(max3(p0.x, p1.x, p2.x), max3(p0.y, p1.y, p2.y), max3(p0.z, p1.z, p2.z), 0.0f);
}

// Walks the input mesh BVH and keeps the first cut along the edge, so an edge gets at most one cut.
__device__ bool findEdgeCut(TetDeviceData& data, const Vec3& p0, const Vec3& p1, float& cutT)
{
    const Ray ray(p0, p1 - p0);
    int stack[64];
    stack[0] = data.triangleBvh.mRootNodes[0];
    int count = 1;
    bool found = false;
    cutT = MaxFloat;

    while (count > 0)
    {
        const int nodeIndex = stack[--count];
        const PackedNodeHalf lower = data.triangleBvh.mNodeLowers[nodeIndex];
        const PackedNodeHalf upper = data.triangleBvh.mNodeUppers[nodeIndex];
        const Bounds3 bounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        float entry;
        float exit;
        if (!header_rayBoundsIntersection(ray, bounds, &entry, &exit) || exit < 0.0f || entry > 1.0f)
            continue;

        if (lower.b)
        {
            const int triangleIndex = static_cast<int>(lower.i);
            const std::uint32_t i0 = data.meshIndices[3 * triangleIndex + 0];
            const std::uint32_t i1 = data.meshIndices[3 * triangleIndex + 1];
            const std::uint32_t i2 = data.meshIndices[3 * triangleIndex + 2];
            if (i0 >= data.meshVertices.size || i1 >= data.meshVertices.size || i2 >= data.meshVertices.size)
                continue;

            float t;
            float u;
            float v;
            if (header_rayTriangleIntersection(
                    ray, data.meshVertices[i0], data.meshVertices[i1], data.meshVertices[i2], t, u, v) &&
                t > 1.0e-6f && t < 1.0f - 1.0e-6f && t < cutT)
            {
                cutT = t;
                found = true;
            }
        }
        else if (count <= 62)
        {
            stack[count++] = static_cast<int>(lower.i);
            stack[count++] = static_cast<int>(upper.i);
        }
    }

    return found;
}

__global__ void createCutVertices(TetDeviceData data, int originalNodeCount, bool countOnly)
{
    CUDA_THREAD_GUARD(edgeIndex, data.numEdges)
    const std::uint64_t edge = data.edges[edgeIndex];
    const int id0 = static_cast<int>(edge >> 32);
    const int id1 = static_cast<int>(edge & 0xffffffffu);

    float cutT;
    const bool cut = findEdgeCut(data, data.nodes[id0], data.nodes[id1], cutT);
    if (countOnly)
    {
        data.firstCutVertex[edgeIndex] = cut ? 1 : 0;
        return;
    }

    if (!cut)
    {
        data.edgeCutVertices[edgeIndex] = -1;
        return;
    }

    const int vertexId = originalNodeCount + data.firstCutVertex[edgeIndex];
    data.nodes[vertexId] = data.nodes[id0] + (data.nodes[id1] - data.nodes[id0]) * cutT;
    data.edgeCutVertices[edgeIndex] = vertexId;
}

__global__ void stampGeomCells(TetDeviceData data)
{
    CUDA_THREAD_GUARD(voxelIndex, data.numVoxels)
    int x, y, z;
    unpackCoords(data.voxels[voxelIndex], x, y, z);
    setCell(data.geomCells.buffer, data, x, y, z);
}

// One 6-neighbor dilate step: dst is solid if src or any face neighbor is solid.
__global__ void dilateGeom(TetDeviceData data, const std::uint32_t* src, std::uint32_t* dst)
{
    CUDA_THREAD_GUARD(cellNr, data.gridNumCells)
    const int x = cellNr / (data.gridNumY * data.gridNumZ);
    const int y = (cellNr / data.gridNumZ) % data.gridNumY;
    const int z = cellNr % data.gridNumZ;

    if (getCell(src, data, x, y, z) || getCell(src, data, x - 1, y, z) || getCell(src, data, x + 1, y, z) ||
        getCell(src, data, x, y - 1, z) || getCell(src, data, x, y + 1, z) || getCell(src, data, x, y, z - 1) ||
        getCell(src, data, x, y, z + 1))
        setCell(dst, data, x, y, z);
}

// One 6-neighbor erode step: dst is solid only if src and all face neighbors are solid.
// Missing neighbors (grid border) count as empty, so the shell shrinks back from the border.
__global__ void erodeGeom(TetDeviceData data, const std::uint32_t* src, std::uint32_t* dst)
{
    CUDA_THREAD_GUARD(cellNr, data.gridNumCells)
    const int x = cellNr / (data.gridNumY * data.gridNumZ);
    const int y = (cellNr / data.gridNumZ) % data.gridNumY;
    const int z = cellNr % data.gridNumZ;

    if (getCell(src, data, x, y, z) && getCell(src, data, x - 1, y, z) && getCell(src, data, x + 1, y, z) &&
        getCell(src, data, x, y - 1, z) && getCell(src, data, x, y + 1, z) && getCell(src, data, x, y, z - 1) &&
        getCell(src, data, x, y, z + 1))
        setCell(dst, data, x, y, z);
}

// One Jacobi step of an exterior flood fill: an empty cell becomes air if it is
// on the grid border or touches an air cell across a face. Repeated until stable,
// every empty cell reachable from outside is air; the rest is enclosed interior.
__global__ void floodAir(TetDeviceData data)
{
    CUDA_THREAD_GUARD(cellNr, data.gridNumCells)
    const int x = cellNr / (data.gridNumY * data.gridNumZ);
    const int y = (cellNr / data.gridNumZ) % data.gridNumY;
    const int z = cellNr % data.gridNumZ;

    if (getCell(data.geomCells.buffer, data, x, y, z) || getCell(data.airCells.buffer, data, x, y, z))
        return;

    const bool onBorder = x == 0 || x == data.gridNumX - 1 || y == 0 || y == data.gridNumY - 1 ||
                          z == 0 || z == data.gridNumZ - 1;
    if (onBorder ||
        getCell(data.airCells.buffer, data, x - 1, y, z) || getCell(data.airCells.buffer, data, x + 1, y, z) ||
        getCell(data.airCells.buffer, data, x, y - 1, z) || getCell(data.airCells.buffer, data, x, y + 1, z) ||
        getCell(data.airCells.buffer, data, x, y, z - 1) || getCell(data.airCells.buffer, data, x, y, z + 1))
    {
        setCell(data.airCells.buffer, data, x, y, z);
        data.anyChanged[0] = 1;
    }
}

// Every cell that is not exterior air is solid (surface or interior). Two passes:
// countOnly to size the output, then a second pass to write the packed coords.
__global__ void collectSolidVoxels(TetDeviceData data, bool countOnly)
{
    CUDA_THREAD_GUARD(cellNr, data.gridNumCells)
    const int x = cellNr / (data.gridNumY * data.gridNumZ);
    const int y = (cellNr / data.gridNumZ) % data.gridNumY;
    const int z = cellNr % data.gridNumZ;

    if (getCell(data.airCells.buffer, data, x, y, z))
        return;

    const int outputIndex = atomicAdd(data.voxelCounter.buffer, 1);
    if (!countOnly)
        data.voxels[outputIndex] = packCoords(x, y, z);
}

template <typename T>
void sortAndUnique(DeviceBuffer<T>& values, int& count)
{
    thrust::device_ptr<T> begin(values.buffer);
    thrust::sort(begin, begin + count);
    const auto end = thrust::unique(begin, begin + count);
    count = static_cast<int>(end - begin);
    values.resize(static_cast<std::size_t>(count), true);
}

int readDeviceInt(const DeviceBuffer<int>& values, int index)
{
    int value = 0;
    cudaCheck(cudaMemcpy(&value, values.buffer + index, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
}

} // namespace

void GpuTetrahedralizer::create(Tetrahedralizer& output, const std::vector<Vec3>& mesh_vertices,
                                const std::vector<std::uint32_t>& mesh_indices,
                                const TetrahedralizerParams& params)
{
    if (!(params.voxelSpacing > 0.0f) || !std::isfinite(params.voxelSpacing))
        throw std::invalid_argument("Voxel spacing must be finite and greater than zero");
    if (params.holeCloseRadius < 0)
        throw std::invalid_argument("Hole close radius must be non-negative");
    if (mesh_indices.size() / 3 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Triangle count exceeds the supported range");

    TetDeviceData data;
    try
    {
        data.numTriangles = static_cast<int>(mesh_indices.size() / 3);
        data.gridSpacing = params.voxelSpacing;
        data.invGridSpacing = 1.0f / data.gridSpacing;

        Bounds3 bounds(Empty);
        for (const Vec3& point : mesh_vertices)
            bounds.include(point);

        // Keep an empty air border after morphological close grows the shell by holeCloseRadius.
        const int borderPad = 1 + params.holeCloseRadius;
        data.worldOrigin =
            bounds.minimum - Vec3(data.gridSpacing, data.gridSpacing, data.gridSpacing) * static_cast<float>(borderPad);

        const Vec3 gridDimensions =
            (bounds.maximum - data.worldOrigin) * data.invGridSpacing +
            Vec3(static_cast<float>(borderPad + 1), static_cast<float>(borderPad + 1),
                 static_cast<float>(borderPad + 1));
        if (gridDimensions.x >= static_cast<float>(kCoordMask) ||
            gridDimensions.y >= static_cast<float>(kCoordMask) ||
            gridDimensions.z >= static_cast<float>(kCoordMask))
            throw std::runtime_error("Voxel grid exceeds the supported coordinate range");

        data.gridNumX = static_cast<int>(std::ceil(gridDimensions.x)) + borderPad;
        data.gridNumY = static_cast<int>(std::ceil(gridDimensions.y)) + borderPad;
        data.gridNumZ = static_cast<int>(std::ceil(gridDimensions.z)) + borderPad;
        const long long gridNumCells =
            static_cast<long long>(data.gridNumX) * data.gridNumY * data.gridNumZ;
        if (gridNumCells > static_cast<long long>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Voxel grid exceeds the supported cell count");
        data.gridNumCells = static_cast<int>(gridNumCells);

        data.meshVertices.set(mesh_vertices);
        data.meshIndices.set(mesh_indices);
        data.firstVoxel.resize(static_cast<std::size_t>(data.numTriangles + 1), false);
        data.firstVoxel.setZero();

        CUDA_LAUNCH(createTriangleVoxels, data.numTriangles, data, true);
        thrust::device_ptr<int> firstVoxel(data.firstVoxel.buffer);
        thrust::exclusive_scan(firstVoxel, firstVoxel + data.numTriangles + 1, firstVoxel);

        const int triangleVoxelCount = readDeviceInt(data.firstVoxel, data.numTriangles);
        if (triangleVoxelCount <= 0)
        {
            data.free();
            return;
        }

        data.voxels.resize(static_cast<std::size_t>(triangleVoxelCount), false);
        CUDA_LAUNCH(createTriangleVoxels, data.numTriangles, data, false);
        data.numVoxels = triangleVoxelCount;
        sortAndUnique(data.voxels, data.numVoxels);

        // Fill the interior: stamp the surface voxels into a dense bit grid, flood the
        // exterior from the border, then keep every cell that the flood did not reach.
        const int numWords = data.gridNumCells / 32 + 1;
        data.geomCells.resize(static_cast<std::size_t>(numWords), false);
        data.geomCells.setZero();
        data.airCells.resize(static_cast<std::size_t>(numWords), false);
        data.airCells.setZero();
        data.anyChanged.resize(1, false);
        data.voxelCounter.resize(1, false);

        CUDA_LAUNCH(stampGeomCells, data.numVoxels, data);

        // Close small holes in the surface shell: dilate then erode by holeCloseRadius.
        if (params.holeCloseRadius > 0)
        {
            data.morphCells.resize(static_cast<std::size_t>(numWords), false);
            for (int step = 0; step < params.holeCloseRadius; ++step)
            {
                data.morphCells.setZero();
                CUDA_LAUNCH(dilateGeom, data.gridNumCells, data, data.geomCells.buffer, data.morphCells.buffer);
                data.geomCells.swap(data.morphCells);
            }
            for (int step = 0; step < params.holeCloseRadius; ++step)
            {
                data.morphCells.setZero();
                CUDA_LAUNCH(erodeGeom, data.gridNumCells, data, data.geomCells.buffer, data.morphCells.buffer);
                data.geomCells.swap(data.morphCells);
            }
            data.morphCells.free();
        }

        const int maxIters = data.gridNumX + data.gridNumY + data.gridNumZ;
        for (int iter = 0; iter < maxIters; ++iter)
        {
            data.anyChanged.setZero();
            CUDA_LAUNCH(floodAir, data.gridNumCells, data);
            if (readDeviceInt(data.anyChanged, 0) == 0)
                break;
        }

        data.voxelCounter.setZero();
        CUDA_LAUNCH(collectSolidVoxels, data.gridNumCells, data, true);
        const int solidVoxelCount = readDeviceInt(data.voxelCounter, 0);
        if (solidVoxelCount <= 0)
        {
            data.free();
            return;
        }

        data.voxels.resize(static_cast<std::size_t>(solidVoxelCount), false);
        data.voxelCounter.setZero();
        CUDA_LAUNCH(collectSolidVoxels, data.gridNumCells, data, false);
        data.numVoxels = solidVoxelCount;

        if (data.numVoxels > std::numeric_limits<int>::max() / 8)
            throw std::runtime_error("Voxel corner count exceeds the supported range");

        int cornerCount = data.numVoxels * 8;
        data.cornerCoords.resize(static_cast<std::size_t>(cornerCount), false);
        CUDA_LAUNCH(createCornerCoords, cornerCount, data);
        sortAndUnique(data.cornerCoords, cornerCount);

        data.numNodes = cornerCount;
        data.nodes.resize(static_cast<std::size_t>(data.numNodes), false);
        CUDA_LAUNCH(createNodes, data.numNodes, data);

        if (data.numVoxels > std::numeric_limits<int>::max() / 5)
            throw std::runtime_error("Tet count exceeds the supported range");
        data.numTets = data.numVoxels * 5;
        data.tetIndices.resize(static_cast<std::size_t>(data.numTets) * 4, false);
        CUDA_LAUNCH(createTets, data.numVoxels, data);

        if (params.cutWithInputMesh)
        {
            if (data.numTets > std::numeric_limits<int>::max() / 6)
                throw std::runtime_error("Tet edge count exceeds the supported range");

            int edgeCount = data.numTets * 6;
            data.edges.resize(static_cast<std::size_t>(edgeCount), false);
            CUDA_LAUNCH(createTetEdges, edgeCount, data);
            sortAndUnique(data.edges, edgeCount);
            data.numEdges = edgeCount;

            data.triangleBoundsLowers.resize(static_cast<std::size_t>(data.numTriangles), false);
            data.triangleBoundsUppers.resize(static_cast<std::size_t>(data.numTriangles), false);
            CUDA_LAUNCH(computeTriangleBounds, data.numTriangles, data);

            BVHBuilderGPU bvhBuilder;
            bvhBuilder.build(data.triangleBvh, data.triangleBoundsLowers.buffer, data.triangleBoundsUppers.buffer,
                             data.numTriangles);

            data.firstCutVertex.resize(static_cast<std::size_t>(data.numEdges + 1), false);
            data.firstCutVertex.setZero();
            CUDA_LAUNCH(createCutVertices, data.numEdges, data, data.numNodes, true);
            thrust::device_ptr<int> firstCutVertex(data.firstCutVertex.buffer);
            thrust::exclusive_scan(firstCutVertex, firstCutVertex + data.numEdges + 1, firstCutVertex);

            const int numCutVertices = readDeviceInt(data.firstCutVertex, data.numEdges);
            data.edgeCutVertices.resize(static_cast<std::size_t>(data.numEdges), false);
            const int originalNodeCount = data.numNodes;
            if (numCutVertices > 0)
            {
                if (numCutVertices > std::numeric_limits<int>::max() - originalNodeCount)
                    throw std::runtime_error("Cut vertex count exceeds the supported range");
                data.nodes.resize(static_cast<std::size_t>(originalNodeCount + numCutVertices), true);
                data.numNodes += numCutVertices;
            }
            CUDA_LAUNCH(createCutVertices, data.numEdges, data, originalNodeCount, false);
        }

        cudaCheck(cudaDeviceSynchronize());

        data.nodes.get(output.nodes);
        data.tetIndices.get(output.tet_indices);
        data.free();
    }
    catch (...)
    {
        data.free();
        throw;
    }
}

} // namespace tetrahedralizer
