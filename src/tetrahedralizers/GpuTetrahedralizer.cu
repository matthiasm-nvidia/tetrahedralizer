#include "GpuTetrahedralizer.h"

#include "tetrahedralizer/Tetrahedralizer.h"
#include "utils/CudaUtils.h"

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

    void free()
    {
        meshVertices.free();
        meshIndices.free();
        voxels.free();
        firstVoxel.free();
        cornerCoords.free();
        nodes.free();
        tetIndices.free();
        numTriangles = 0;
        numVoxels = 0;
        numNodes = 0;
        numTets = 0;
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
        data.worldOrigin = bounds.minimum - Vec3(data.gridSpacing, data.gridSpacing, data.gridSpacing);

        const Vec3 gridDimensions = (bounds.maximum - data.worldOrigin) * data.invGridSpacing + Vec3(2.0f, 2.0f, 2.0f);
        if (gridDimensions.x >= static_cast<float>(kCoordMask) ||
            gridDimensions.y >= static_cast<float>(kCoordMask) ||
            gridDimensions.z >= static_cast<float>(kCoordMask))
            throw std::runtime_error("Voxel grid exceeds the supported coordinate range");

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
