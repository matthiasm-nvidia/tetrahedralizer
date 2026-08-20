#pragma once

#include "TetCutTemplates.h"

#include "tetrahedralizer/Tetrahedralizer.h"
#include "utils/CudaUtils.h"
#include "utils/GpuBVH.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace tetrahedralizer
{

constexpr std::uint64_t kCoordMask = (1ull << 21) - 1ull;

// Cell overlap margin as a fraction of the voxel spacing. Without it, a triangle
// that grazes a cell can be rejected by floating point error and leave a pinhole
// in the surface shell, which lets the exterior flood fill leak into the interior.
constexpr float kVoxelOverlapMargin = 1.0e-3f;

static __device__ __constant__ int kTetEdges[6][2] = {
    {tet_cut::kTetEdges[0][0], tet_cut::kTetEdges[0][1]},
    {tet_cut::kTetEdges[1][0], tet_cut::kTetEdges[1][1]},
    {tet_cut::kTetEdges[2][0], tet_cut::kTetEdges[2][1]},
    {tet_cut::kTetEdges[3][0], tet_cut::kTetEdges[3][1]},
    {tet_cut::kTetEdges[4][0], tet_cut::kTetEdges[4][1]},
    {tet_cut::kTetEdges[5][0], tet_cut::kTetEdges[5][1]},
};
static __device__ __constant__ int kTetFaces[4][3] = {
    {tet_cut::kTetFaces[0][0], tet_cut::kTetFaces[0][1], tet_cut::kTetFaces[0][2]},
    {tet_cut::kTetFaces[1][0], tet_cut::kTetFaces[1][1], tet_cut::kTetFaces[1][2]},
    {tet_cut::kTetFaces[2][0], tet_cut::kTetFaces[2][1], tet_cut::kTetFaces[2][2]},
    {tet_cut::kTetFaces[3][0], tet_cut::kTetFaces[3][1], tet_cut::kTetFaces[3][2]},
};

__host__ __device__ inline std::uint64_t packCoords(int x, int y, int z)
{
    return (static_cast<std::uint64_t>(x) << 42) |
           (static_cast<std::uint64_t>(y) << 21) |
           static_cast<std::uint64_t>(z);
}

__host__ __device__ inline void unpackCoords(std::uint64_t packed, int& x, int& y, int& z)
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

    // Input mesh
    DeviceBuffer<Vec3> meshVertices;
    DeviceBuffer<std::uint32_t> meshIndices;
    DeviceBuffer<Vec4> triangleBoundsLowers;
    DeviceBuffer<Vec4> triangleBoundsUppers;
    GpuBVH triangleBvh;

    // Voxel grid
    DeviceBuffer<std::uint64_t> voxels;
    DeviceBuffer<int> firstVoxel;
    DeviceBuffer<std::uint32_t> geomCells;
    DeviceBuffer<std::uint32_t> airCells;
    DeviceBuffer<std::uint32_t> morphCells;
    DeviceBuffer<int> voxelCounter;

    // Voxel-split nodes
    DeviceBuffer<int> connectedVoxelFaces;
    DeviceBuffer<int> mergedCornerIds;
    DeviceBuffer<int> nodeOffsets;
    DeviceBuffer<int> voxelCornerNodes;

    // Tet mesh
    DeviceBuffer<Vec3> nodes;
    DeviceBuffer<int> tetIndices;
    DeviceBuffer<int> tetNeighbors; // 4 per tet: adjacent tet index or -1
    DeviceBuffer<std::uint64_t> edges;

    // Subdivision / boundary cuts
    DeviceBuffer<int> firstCutVertex;
    DeviceBuffer<int> edgeCutVertices;
    DeviceBuffer<int> firstSteiner;
    DeviceBuffer<int> steinerVertexId;
    DeviceBuffer<int> firstNewTet;
    DeviceBuffer<tet_cut::CutTemplateTables> cutTables;

    // Edge collapse
    DeviceBuffer<int> collapseMap;
    DeviceBuffer<Vec3> collapsePositions;

    // Optimize / project
    DeviceBuffer<int> anyChanged;
    DeviceBuffer<Vec3> smoothOffsets;
    DeviceBuffer<int> smoothCounts;
    DeviceBuffer<Vec3> moveOffsets;
    DeviceBuffer<float> moveScales;
    DeviceBuffer<int> moveBlocked;

    void free()
    {
        meshVertices.free();
        meshIndices.free();
        triangleBoundsLowers.free();
        triangleBoundsUppers.free();
        triangleBvh.free();
        voxels.free();
        firstVoxel.free();
        geomCells.free();
        airCells.free();
        morphCells.free();
        voxelCounter.free();
        connectedVoxelFaces.free();
        mergedCornerIds.free();
        nodeOffsets.free();
        voxelCornerNodes.free();
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
        collapseMap.free();
        collapsePositions.free();
        anyChanged.free();
        smoothOffsets.free();
        smoothCounts.free();
        moveOffsets.free();
        moveScales.free();
        moveBlocked.free();
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

__device__ inline int cellIndex(const TetDeviceData& data, int x, int y, int z)
{
    return (x * data.gridNumY + y) * data.gridNumZ + z;
}

__device__ inline bool getCell(const std::uint32_t* cells, const TetDeviceData& data, int x, int y, int z)
{
    if (x < 0 || x >= data.gridNumX || y < 0 || y >= data.gridNumY || z < 0 || z >= data.gridNumZ)
        return false;
    const int index = cellIndex(data, x, y, z);
    return (cells[index >> 5] & (1u << (index & 31))) != 0;
}

__device__ inline void setCell(std::uint32_t* cells, const TetDeviceData& data, int x, int y, int z)
{
    const int index = cellIndex(data, x, y, z);
    atomicOr(&cells[index >> 5], 1u << (index & 31));
}

__device__ inline int findCoord(const DeviceBuffer<std::uint64_t>& coords, std::uint64_t target)
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

inline int readDeviceInt(const DeviceBuffer<int>& values, int index)
{
    int value = 0;
    cudaCheck(cudaMemcpy(&value, values.buffer + index, sizeof(int), cudaMemcpyDeviceToHost));
    return value;
}

template <typename T>
inline void sortAndUnique(DeviceBuffer<T>& values, int& count)
{
    thrust::device_ptr<T> begin(values.buffer);
    thrust::sort(begin, begin + count);
    const auto end = thrust::unique(begin, begin + count);
    count = static_cast<int>(end - begin);
    values.resize(static_cast<std::size_t>(count), true);
}

void voxelizeTriangles(TetDeviceData& data);
void fillInterior(TetDeviceData& data, int holeCloseRadius);
void buildInputMeshBvh(TetDeviceData& data);
void createSplitVoxelNodes(TetDeviceData& data);
void createFiveTets(TetDeviceData& data);
void subdivideLongEdges(TetDeviceData& data, float maxEdgeLength);
void collapseShortEdges(TetDeviceData& data, float minEdgeLength);
bool computeNeighbors(TetDeviceData& data);
void computeSurfaceNormals(TetDeviceData& data, DeviceBuffer<Vec3>& normals);
void separateBoundaryFaces(TetDeviceData& data);
void runOptimization(TetDeviceData& data, const TetrahedralizerParams& params);
void uploadMeshFromHost(TetDeviceData& data, const Tetrahedralizer& mesh);
void downloadMeshToHost(TetDeviceData& data, Tetrahedralizer& mesh);

} // namespace tetrahedralizer
