#include "GpuTetrahedralizer.h"
#include "TetCutTemplates.h"

#include "tetrahedralizer/Tetrahedralizer.h"
#include "utils/CudaUtils.h"
#include "utils/Geometry.h"
#include "utils/GpuBVH.h"
#include "utils/Math.h"

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

// Cell overlap margin as a fraction of the voxel spacing. Without it, a triangle
// that grazes a cell can be rejected by floating point error and leave a pinhole
// in the surface shell, which lets the exterior flood fill leak into the interior.
constexpr float kVoxelOverlapMargin = 1.0e-3f;

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

// Edge order / mask bits match tet_cut templates:
// bit0=01, bit1=12, bit2=20, bit3=03, bit4=13, bit5=23
__device__ __constant__ int kTetEdges[6][2] = {
    {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3},
};

// Local face corners; opposite vertex is the missing one of {0,1,2,3}.
__device__ __constant__ int kTetFaces[4][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
};

struct Face
{
    int id0 = 0;
    int id1 = 0;
    int id2 = 0;
    int faceNr = 0; // 4 * tetIndex + localFace

    __host__ __device__ bool operator==(const Face& other) const
    {
        return id0 == other.id0 && id1 == other.id1 && id2 == other.id2;
    }
};

struct FaceComparator
{
    __host__ __device__ bool operator()(const Face& a, const Face& b) const
    {
        if (a.id0 != b.id0)
            return a.id0 < b.id0;
        if (a.id1 != b.id1)
            return a.id1 < b.id1;
        return a.id2 < b.id2;
    }
};

__device__ void sort3(int& a, int& b, int& c)
{
    if (a > b)
    {
        const int tmp = a;
        a = b;
        b = tmp;
    }
    if (b > c)
    {
        const int tmp = b;
        b = c;
        c = tmp;
    }
    if (a > b)
    {
        const int tmp = a;
        a = b;
        b = tmp;
    }
}

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
    DeviceBuffer<int> tetNeighbors; // 4 per tet: adjacent tet index or -1
    DeviceBuffer<std::uint64_t> edges;
    DeviceBuffer<int> firstCutVertex;
    DeviceBuffer<int> edgeCutVertices;
    DeviceBuffer<int> firstSteiner;
    DeviceBuffer<int> steinerVertexId;
    DeviceBuffer<int> firstNewTet;
    DeviceBuffer<tet_cut::CutTemplateTables> cutTables;
    DeviceBuffer<Vec4> triangleBoundsLowers;
    DeviceBuffer<Vec4> triangleBoundsUppers;
    GpuBVH triangleBvh;
    DeviceBuffer<std::uint32_t> geomCells;
    DeviceBuffer<std::uint32_t> airCells;
    DeviceBuffer<std::uint32_t> morphCells;
    DeviceBuffer<int> anyChanged;
    DeviceBuffer<int> voxelCounter;
    DeviceBuffer<Vec3> smoothOffsets;
    DeviceBuffer<int> smoothCounts;

    void free()
    {
        meshVertices.free();
        meshIndices.free();
        voxels.free();
        firstVoxel.free();
        cornerCoords.free();
        nodes.free();
        tetIndices.free();
        tetNeighbors.free();
        edges.free();
        firstCutVertex.free();
        edgeCutVertices.free();
        firstSteiner.free();
        steinerVertexId.free();
        firstNewTet.free();
        cutTables.free();
        triangleBoundsLowers.free();
        triangleBoundsUppers.free();
        triangleBvh.free();
        geomCells.free();
        airCells.free();
        morphCells.free();
        anyChanged.free();
        voxelCounter.free();
        smoothOffsets.free();
        smoothCounts.free();
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

    const float margin = data.gridSpacing * kVoxelOverlapMargin;
    const int x0 = static_cast<int>(floorf((min3(p0.x, p1.x, p2.x) - margin - origin.x) * data.invGridSpacing));
    const int y0 = static_cast<int>(floorf((min3(p0.y, p1.y, p2.y) - margin - origin.y) * data.invGridSpacing));
    const int z0 = static_cast<int>(floorf((min3(p0.z, p1.z, p2.z) - margin - origin.z) * data.invGridSpacing));
    const int x1 = static_cast<int>(floorf((max3(p0.x, p1.x, p2.x) + margin - origin.x) * data.invGridSpacing));
    const int y1 = static_cast<int>(floorf((max3(p0.y, p1.y, p2.y) + margin - origin.y) * data.invGridSpacing));
    const int z1 = static_cast<int>(floorf((max3(p0.z, p1.z, p2.z) + margin - origin.z) * data.invGridSpacing));

    int count = 0;
    int outputIndex = data.numVoxels + data.firstVoxel[triangleIndex];
    const float halfExtent = data.gridSpacing * 0.5f + margin;
    const DeviceVec3 halfExtents(halfExtent, halfExtent, halfExtent);
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

// Six times the volume of the regular tet returned by regularTetCorners.
constexpr float kRegularSixVolume = 16.0f;

// Regular tet centered at the origin with equal edge lengths and positive orientation.
__device__ void regularTetCorners(Vec3 q[4])
{
    q[0] = Vec3(1.0f, 1.0f, 1.0f);
    q[1] = Vec3(1.0f, -1.0f, -1.0f);
    q[2] = Vec3(-1.0f, -1.0f, 1.0f);
    q[3] = Vec3(-1.0f, 1.0f, -1.0f);
}

// Fit a rotated regular tet with volumeFactor * the current tet volume and accumulate
// the corner deltas.
__global__ void smoothAccumulate(TetDeviceData data, float volumeFactor)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    const int id0 = data.tetIndices[4 * tetIndex + 0];
    const int id1 = data.tetIndices[4 * tetIndex + 1];
    const int id2 = data.tetIndices[4 * tetIndex + 2];
    const int id3 = data.tetIndices[4 * tetIndex + 3];

    const Vec3 p0 = data.nodes[id0];
    const Vec3 p1 = data.nodes[id1];
    const Vec3 p2 = data.nodes[id2];
    const Vec3 p3 = data.nodes[id3];

    const Mat33 P(p1 - p0, p2 - p0, p3 - p0);
    const float sixVolume = P.getDeterminant();
    if (!(sixVolume > 1.0e-12f))
        return;

    Vec3 q[4];
    regularTetCorners(q);
    const Mat33 Q(q[1] - q[0], q[2] - q[0], q[3] - q[0]);
    const float targetScale = cbrtf(volumeFactor * sixVolume / kRegularSixVolume);

    Mat33 R, U, D;
    headerPolarDecomposition(P * Q.getInverse(), R, U, D);

    const Vec3 center = (p0 + p1 + p2 + p3) * 0.25f;
    const Vec3 targets[4] = {
        center + R * (q[0] * targetScale),
        center + R * (q[1] * targetScale),
        center + R * (q[2] * targetScale),
        center + R * (q[3] * targetScale),
    };
    const int ids[4] = {id0, id1, id2, id3};
    const Vec3 positions[4] = {p0, p1, p2, p3};
    for (int corner = 0; corner < 4; ++corner)
    {
        AtomicAdd(data.smoothOffsets.buffer + ids[corner], targets[corner] - positions[corner]);
        AtomicAdd(data.smoothCounts.buffer + ids[corner], 1);
    }
}

__global__ void smoothApply(TetDeviceData data)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    const int count = data.smoothCounts[nodeIndex];
    if (count == 0)
        return;
    data.nodes[nodeIndex] += data.smoothOffsets[nodeIndex] / static_cast<float>(count);
}

void smoothTets(TetDeviceData& data, int iterations, float volumeFactor)
{
    if (iterations <= 0 || data.numTets <= 0 || data.numNodes <= 0)
        return;

    data.smoothOffsets.resize(static_cast<std::size_t>(data.numNodes));
    data.smoothCounts.resize(static_cast<std::size_t>(data.numNodes));
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        data.smoothOffsets.setZero();
        data.smoothCounts.setZero();
        CUDA_LAUNCH(smoothAccumulate, data.numTets, data, volumeFactor);
        CUDA_LAUNCH(smoothApply, data.numNodes, data);
    }
    cudaCheck(cudaDeviceSynchronize());
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

__global__ void fillFaces(TetDeviceData data, Face* faces)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    for (int face = 0; face < 4; ++face)
    {
        int id0 = data.tetIndices[4 * tetIndex + kTetFaces[face][0]];
        int id1 = data.tetIndices[4 * tetIndex + kTetFaces[face][1]];
        int id2 = data.tetIndices[4 * tetIndex + kTetFaces[face][2]];
        sort3(id0, id1, id2);

        const int faceNr = 4 * tetIndex + face;
        faces[faceNr] = Face{id0, id1, id2, faceNr};
        data.tetNeighbors[faceNr] = -1;
    }
}

// Pairs faces that share the same three vertices. More than two tets on one face
// increments nonManifold (same pattern as triangle edge neighbors in mesh-tools-lib).
__global__ void fillNeighbors(int numFaces, const Face* faces, int* neighbors, int* nonManifold)
{
    CUDA_THREAD_GUARD(faceIndex, numFaces)

    const Face face0 = faces[faceIndex];
    if (faceIndex > 0 && face0 == faces[faceIndex - 1])
        return;

    int num = 0;
    int next = faceIndex + 1;
    while (next < numFaces && face0 == faces[next])
    {
        const Face face1 = faces[next];
        neighbors[face0.faceNr] = face1.faceNr / 4;
        neighbors[face1.faceNr] = face0.faceNr / 4;
        ++num;
        ++next;
    }

    if (num > 1)
        atomicAdd(nonManifold, 1);
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

// Cut vertices stay at least this fraction of the edge away from both end nodes.
constexpr float kMinCutT = 0.1f;

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

    // Keep the cut away from the edge ends so the child tets do not get degenerate.
    cutT = fminf(fmaxf(cutT, kMinCutT), 1.0f - kMinCutT);

    const int vertexId = originalNodeCount + data.firstCutVertex[edgeIndex];
    data.nodes[vertexId] = data.nodes[id0] + (data.nodes[id1] - data.nodes[id0]) * cutT;
    data.edgeCutVertices[edgeIndex] = vertexId;
}

// Midpoint cuts for edges marked in edgeCutFlags (1 = cut, 0 = skip). Used by tests.
__global__ void createMarkedCutVertices(TetDeviceData data, const int* edgeCutFlags, int originalNodeCount,
                                        bool countOnly)
{
    CUDA_THREAD_GUARD(edgeIndex, data.numEdges)
    const bool cut = edgeCutFlags[edgeIndex] != 0;
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

    const std::uint64_t edge = data.edges[edgeIndex];
    const int id0 = static_cast<int>(edge >> 32);
    const int id1 = static_cast<int>(edge & 0xffffffffu);
    const int vertexId = originalNodeCount + data.firstCutVertex[edgeIndex];
    data.nodes[vertexId] = (data.nodes[id0] + data.nodes[id1]) * 0.5f;
    data.edgeCutVertices[edgeIndex] = vertexId;
}

__device__ int findPackedEdge(const DeviceBuffer<std::uint64_t>& edges, std::uint64_t target)
{
    int first = 0;
    int last = static_cast<int>(edges.size);
    while (first < last)
    {
        const int middle = first + (last - first) / 2;
        if (edges.buffer[middle] < target)
            first = middle + 1;
        else
            last = middle;
    }
    if (first >= static_cast<int>(edges.size) || edges.buffer[first] != target)
        return -1;
    return first;
}

// Fills local[0..9]: corners and edge cut verts (-1 if uncut). Returns cut mask.
__device__ int resolveTetCutLocals(const TetDeviceData& data, int tetIndex, int local[10])
{
    for (int i = 0; i < 4; ++i)
        local[i] = data.tetIndices.buffer[4 * tetIndex + i];

    int mask = 0;
    for (int e = 0; e < 6; ++e)
    {
        const int id0 = local[kTetEdges[e][0]];
        const int id1 = local[kTetEdges[e][1]];
        const int edgeIndex = findPackedEdge(data.edges, packEdge(id0, id1));
        const int cutId = edgeIndex >= 0 ? data.edgeCutVertices.buffer[edgeIndex] : -1;
        local[4 + e] = cutId;
        if (cutId >= 0)
            mask |= 1 << e;
    }
    return mask;
}

__device__ int resolveDiagBits(const tet_cut::CutTemplateTables& tables, int mask, const int local[10])
{
    int diagBits = 0;
    for (int f = 0; f < 4; ++f)
    {
        const int a = tables.diagA[mask][f];
        const int b = tables.diagB[mask][f];
        if (a < 0)
            continue;
        if (local[a] > local[b])
            diagBits |= 1 << f;
    }
    return diagBits;
}

void uploadCutTables(TetDeviceData& data)
{
    tet_cut::CutTemplateTables hostTables;
    tet_cut::buildCutTemplateTables(hostTables);
    if (hostTables.childCount[0][0] != 1 || hostTables.childCount[1][0] != 2 ||
        hostTables.childCount[63][0] != 8)
        throw std::runtime_error("Tet cut template tables failed sanity checks");
    for (int mask = 0; mask < 64; ++mask)
    {
        for (int diagBits = 0; diagBits < 16; ++diagBits)
        {
            if (hostTables.childCount[mask][diagBits] > tet_cut::kMaxChildren)
                throw std::runtime_error("Tet cut template exceeds max child count");
        }
    }
    data.cutTables.resize(1, false);
    cudaCheck(cudaMemcpy(data.cutTables.buffer, &hostTables, sizeof(hostTables), cudaMemcpyHostToDevice));
}

__global__ void createSteinerVertices(TetDeviceData data, int originalNodeCount, bool countOnly)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int local[10];
    const int mask = resolveTetCutLocals(data, tetIndex, local);
    const tet_cut::CutTemplateTables& tables = data.cutTables[0];
    const int diagBits = resolveDiagBits(tables, mask, local);
    const bool needs = tables.needsSteiner[mask][diagBits] != 0;

    if (countOnly)
    {
        data.firstSteiner[tetIndex] = needs ? 1 : 0;
        return;
    }

    if (!needs)
    {
        data.steinerVertexId[tetIndex] = -1;
        return;
    }

    const int vertexId = originalNodeCount + data.firstSteiner[tetIndex];
    data.nodes[vertexId] =
        (data.nodes[local[0]] + data.nodes[local[1]] + data.nodes[local[2]] + data.nodes[local[3]]) * 0.25f;
    data.steinerVertexId[tetIndex] = vertexId;
}

__global__ void splitTets(TetDeviceData data, bool countOnly)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int local[11];
    const int mask = resolveTetCutLocals(data, tetIndex, local);
    const tet_cut::CutTemplateTables& tables = data.cutTables[0];
    const int diagBits = resolveDiagBits(tables, mask, local);
    const int childCount = tables.childCount[mask][diagBits];

    if (countOnly)
    {
        data.firstNewTet[tetIndex] = childCount > 0 ? childCount - 1 : 0;
        return;
    }

    if (mask == 0 || childCount <= 1)
        return;

    local[10] = data.steinerVertexId[tetIndex];

    const int firstExtra = data.numTets + data.firstNewTet[tetIndex];
    for (int child = 0; child < childCount; ++child)
    {
        const int outTet = child == 0 ? tetIndex : firstExtra + (child - 1);
        for (int corner = 0; corner < 4; ++corner)
        {
            const int localId = tables.children[mask][diagBits][child * 4 + corner];
            data.tetIndices[4 * outTet + corner] = local[localId];
        }
    }
}

// Classify each tet from the signed side of the closest input triangle.
// Consistently outward-facing input triangles make inside=true.
__global__ void markKeptTets(TetDeviceData data, int* keptTetOffsets)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    Vec3 center(Zero);
    for (int corner = 0; corner < 4; ++corner)
        center += data.nodes[data.tetIndices[4 * tetIndex + corner]];
    center *= 0.25f;

    int closestTri = -1;
    Vec3 bary(Zero);
    Vec3 closestPos(Zero);
    bool inside = false;
    header_queryClosestPoint(data.triangleBvh, center, 0.0f, data.meshVertices.buffer,
                             reinterpret_cast<int*>(data.meshIndices.buffer), closestTri, bary, closestPos, inside);

    // An inconclusive query keeps the tet instead of deleting geometry.
    keptTetOffsets[tetIndex] = closestTri < 0 || inside ? 1 : 0;
}

__global__ void compactKeptTets(TetDeviceData data, const int* keptTetOffsets, int* compactedTetIndices)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    if (keptTetOffsets[tetIndex + 1] == keptTetOffsets[tetIndex])
        return;

    const int outputTet = keptTetOffsets[tetIndex];
    for (int corner = 0; corner < 4; ++corner)
        compactedTetIndices[4 * outputTet + corner] = data.tetIndices[4 * tetIndex + corner];
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

// Builds tetNeighbors[4 * tet + face] = adjacent tet, or -1 on the boundary.
// Returns true if every interior face is shared by exactly two tets.
bool computeNeighbors(TetDeviceData& data)
{
    if (data.numTets <= 0)
    {
        data.tetNeighbors.free();
        return true;
    }

    if (data.numTets > std::numeric_limits<int>::max() / 4)
        throw std::runtime_error("Tet face count exceeds the supported range");

    const int numFaces = data.numTets * 4;
    data.tetNeighbors.resize(static_cast<std::size_t>(numFaces), false);

    DeviceBuffer<Face> faces;
    faces.resize(static_cast<std::size_t>(numFaces), false);
    CUDA_LAUNCH(fillFaces, data.numTets, data, faces.buffer);

    thrust::device_ptr<Face> facePtr(faces.buffer);
    thrust::sort(facePtr, facePtr + numFaces, FaceComparator());

    DeviceBuffer<int> nonManifold;
    nonManifold.resize(1, false);
    nonManifold.setZero();
    CUDA_LAUNCH(fillNeighbors, numFaces, numFaces, faces.buffer, data.tetNeighbors.buffer, nonManifold.buffer);

    const bool manifold = readDeviceInt(nonManifold, 0) == 0;
    faces.free();
    nonManifold.free();
    return manifold;
}

void buildUniqueTetEdges(TetDeviceData& data)
{
    if (data.numTets > std::numeric_limits<int>::max() / 6)
        throw std::runtime_error("Tet edge count exceeds the supported range");

    int edgeCount = data.numTets * 6;
    data.edges.resize(static_cast<std::size_t>(edgeCount), false);
    CUDA_LAUNCH(createTetEdges, edgeCount, data);
    sortAndUnique(data.edges, edgeCount);
    data.numEdges = edgeCount;
}

// Assumes edges + edgeCutVertices are already filled for all cut edges.
void applyCutTemplates(TetDeviceData& data)
{
    uploadCutTables(data);

    data.firstSteiner.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstSteiner.setZero();
    CUDA_LAUNCH(createSteinerVertices, data.numTets, data, data.numNodes, true);
    thrust::device_ptr<int> firstSteiner(data.firstSteiner.buffer);
    thrust::exclusive_scan(firstSteiner, firstSteiner + data.numTets + 1, firstSteiner);

    const int numSteiner = readDeviceInt(data.firstSteiner, data.numTets);
    data.steinerVertexId.resize(static_cast<std::size_t>(data.numTets), false);
    const int nodesBeforeSteiner = data.numNodes;
    if (numSteiner > 0)
    {
        if (numSteiner > std::numeric_limits<int>::max() - nodesBeforeSteiner)
            throw std::runtime_error("Steiner vertex count exceeds the supported range");
        data.nodes.resize(static_cast<std::size_t>(nodesBeforeSteiner + numSteiner), true);
        data.numNodes += numSteiner;
    }
    CUDA_LAUNCH(createSteinerVertices, data.numTets, data, nodesBeforeSteiner, false);

    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstNewTet.setZero();
    CUDA_LAUNCH(splitTets, data.numTets, data, true);
    thrust::device_ptr<int> firstNewTet(data.firstNewTet.buffer);
    thrust::exclusive_scan(firstNewTet, firstNewTet + data.numTets + 1, firstNewTet);

    const int numExtraTets = readDeviceInt(data.firstNewTet, data.numTets);
    if (numExtraTets > 0)
    {
        if (numExtraTets > std::numeric_limits<int>::max() - data.numTets)
            throw std::runtime_error("Cut tet count exceeds the supported range");
        data.tetIndices.resize(static_cast<std::size_t>(data.numTets + numExtraTets) * 4, true);
    }
    CUDA_LAUNCH(splitTets, data.numTets, data, false);
    data.numTets += numExtraTets;
}

void carveOutsideTets(TetDeviceData& data)
{
    if (data.numTets <= 0)
        return;

    DeviceBuffer<int> keptTetOffsets;
    keptTetOffsets.resize(static_cast<std::size_t>(data.numTets + 1), false);
    keptTetOffsets.setZero();
    CUDA_LAUNCH(markKeptTets, data.numTets, data, keptTetOffsets.buffer);

    thrust::device_ptr<int> offsets(keptTetOffsets.buffer);
    thrust::exclusive_scan(offsets, offsets + data.numTets + 1, offsets);
    const int keptTetCount = readDeviceInt(keptTetOffsets, data.numTets);

    if (keptTetCount < data.numTets)
    {
        DeviceBuffer<int> compactedTetIndices;
        compactedTetIndices.resize(static_cast<std::size_t>(keptTetCount) * 4, false);
        CUDA_LAUNCH(compactKeptTets, data.numTets, data, keptTetOffsets.buffer, compactedTetIndices.buffer);
        data.tetIndices.swap(compactedTetIndices);
        compactedTetIndices.free();
        data.numTets = keptTetCount;
    }

    keptTetOffsets.free();
}

void applyMarkedEdgeCuts(TetDeviceData& data, const std::vector<int>& edgeCutFlags)
{
    if (static_cast<int>(edgeCutFlags.size()) != data.numEdges)
        throw std::invalid_argument("Edge cut flag count must match unique edge count");

    DeviceBuffer<int> flags;
    flags.set(edgeCutFlags);

    data.firstCutVertex.resize(static_cast<std::size_t>(data.numEdges + 1), false);
    data.firstCutVertex.setZero();
    CUDA_LAUNCH(createMarkedCutVertices, data.numEdges, data, flags.buffer, data.numNodes, true);
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
    CUDA_LAUNCH(createMarkedCutVertices, data.numEdges, data, flags.buffer, originalNodeCount, false);
    flags.free();

    if (numCutVertices > 0)
        applyCutTemplates(data);
}

void uploadMeshFromHost(TetDeviceData& data, const Tetrahedralizer& mesh)
{
    if (mesh.nodes.empty() || mesh.tet_indices.size() < 4)
        throw std::invalid_argument("Mesh must contain at least one tetrahedron");

    data.nodes.set(mesh.nodes);
    data.tetIndices.set(mesh.tet_indices);
    data.numNodes = static_cast<int>(mesh.nodes.size());
    data.numTets = static_cast<int>(mesh.tet_indices.size() / 4);
}

void downloadMeshToHost(TetDeviceData& data, Tetrahedralizer& mesh)
{
    data.nodes.get(mesh.nodes);
    data.tetIndices.get(mesh.tet_indices);
    data.tetNeighbors.get(mesh.tet_neighbors);
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
    if (params.numSmoothingIterations < 0)
        throw std::invalid_argument("Smoothing iteration count must be non-negative");
    if (!(params.volumeFactor > 0.0f) || !std::isfinite(params.volumeFactor))
        throw std::invalid_argument("Volume factor must be finite and greater than zero");
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

        smoothTets(data, params.numSmoothingIterations, params.volumeFactor);

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

            if (numCutVertices > 0)
                applyCutTemplates(data);

            carveOutsideTets(data);
        }

        cudaCheck(cudaDeviceSynchronize());

        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after tetrahedralization");

        data.nodes.get(output.nodes);
        data.tetIndices.get(output.tet_indices);
        data.tetNeighbors.get(output.tet_neighbors);
        data.free();
    }
    catch (...)
    {
        data.free();
        throw;
    }
}

void GpuTetrahedralizer::cutRandomEdges(Tetrahedralizer& mesh, float probability, unsigned seed)
{
    if (!(probability >= 0.0f) || !(probability <= 1.0f) || !std::isfinite(probability))
        throw std::invalid_argument("Cut probability must be in [0, 1]");

    TetDeviceData data;
    try
    {
        uploadMeshFromHost(data, mesh);
        buildUniqueTetEdges(data);

        std::vector<int> flags(static_cast<std::size_t>(data.numEdges), 0);
        unsigned state = seed ? seed : 1u;
        for (int i = 0; i < data.numEdges; ++i)
        {
            state = state * 1664525u + 1013904223u;
            const float u = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
            flags[static_cast<std::size_t>(i)] = u < probability ? 1 : 0;
        }

        applyMarkedEdgeCuts(data, flags);

        cudaCheck(cudaDeviceSynchronize());
        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after random cuts");

        downloadMeshToHost(data, mesh);
        data.free();
    }
    catch (...)
    {
        data.free();
        throw;
    }
}

void GpuTetrahedralizer::cutSingleTetByMask(Tetrahedralizer& mesh, int mask)
{
    if (mask < 0 || mask > 63)
        throw std::invalid_argument("Edge-cut mask must be in [0, 63]");
    if (mesh.numTets() != 1 || mesh.tet_indices.size() != 4 || mesh.nodes.size() < 4)
        throw std::invalid_argument("cutSingleTetByMask requires a mesh with exactly one tetrahedron");

    // Same edge order as tet_cut / device kTetEdges.
    static constexpr int kEdges[6][2] = {
        {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3},
    };

    TetDeviceData data;
    try
    {
        uploadMeshFromHost(data, mesh);
        buildUniqueTetEdges(data);

        std::vector<std::uint64_t> hostEdges;
        data.edges.get(hostEdges);

        std::vector<int> flags(static_cast<std::size_t>(data.numEdges), 0);
        const int* ids = mesh.tet_indices.data();
        for (int e = 0; e < 6; ++e)
        {
            if (((mask >> e) & 1) == 0)
                continue;

            const std::uint64_t packed = packEdge(ids[kEdges[e][0]], ids[kEdges[e][1]]);
            const auto it = std::lower_bound(hostEdges.begin(), hostEdges.end(), packed);
            if (it == hostEdges.end() || *it != packed)
                throw std::runtime_error("Failed to map single-tet edge mask to unique edges");
            flags[static_cast<std::size_t>(it - hostEdges.begin())] = 1;
        }

        applyMarkedEdgeCuts(data, flags);

        cudaCheck(cudaDeviceSynchronize());
        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after mask cut");

        downloadMeshToHost(data, mesh);
        data.free();
    }
    catch (...)
    {
        data.free();
        throw;
    }
}

} // namespace tetrahedralizer
