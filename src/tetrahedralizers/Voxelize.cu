#include "TetDeviceData.h"

#include "utils/Geometry.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace tetrahedralizer
{
namespace
{

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

    const Vec3 p0 = data.meshVertices[i0];
    const Vec3 p1 = data.meshVertices[i1];
    const Vec3 p2 = data.meshVertices[i2];
    const Vec3 origin = data.worldOrigin;

    const float margin = data.gridSpacing * kVoxelOverlapMargin;
    const int x0 = static_cast<int>(floorf((Min(p0.x, p1.x, p2.x) - margin - origin.x) * data.invGridSpacing));
    const int y0 = static_cast<int>(floorf((Min(p0.y, p1.y, p2.y) - margin - origin.y) * data.invGridSpacing));
    const int z0 = static_cast<int>(floorf((Min(p0.z, p1.z, p2.z) - margin - origin.z) * data.invGridSpacing));
    const int x1 = static_cast<int>(floorf((Max(p0.x, p1.x, p2.x) + margin - origin.x) * data.invGridSpacing));
    const int y1 = static_cast<int>(floorf((Max(p0.y, p1.y, p2.y) + margin - origin.y) * data.invGridSpacing));
    const int z1 = static_cast<int>(floorf((Max(p0.z, p1.z, p2.z) + margin - origin.z) * data.invGridSpacing));

    int count = 0;
    int outputIndex = data.numVoxels + data.firstVoxel[triangleIndex];
    const float halfExtent = data.gridSpacing * 0.5f + margin;
    const Vec3 halfExtents(halfExtent, halfExtent, halfExtent);
    for (int x = x0; x <= x1; ++x)
    {
        for (int y = y0; y <= y1; ++y)
        {
            for (int z = z0; z <= z1; ++z)
            {
                const Vec3 center(
                    origin.x + (static_cast<float>(x) + 0.5f) * data.gridSpacing,
                    origin.y + (static_cast<float>(y) + 0.5f) * data.gridSpacing,
                    origin.z + (static_cast<float>(z) + 0.5f) * data.gridSpacing);
                if (!header_boxTriangleIntersection(p0, p1, p2, center, halfExtents))
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

} // namespace

void voxelizeTriangles(TetDeviceData& data)
{
    CUDA_LAUNCH(createTriangleVoxels, data.numTriangles, data, true);
    thrust::device_ptr<int> firstVoxel(data.firstVoxel.buffer);
    thrust::exclusive_scan(firstVoxel, firstVoxel + data.numTriangles + 1, firstVoxel);

    const int triangleVoxelCount = readDeviceInt(data.firstVoxel, data.numTriangles);
    if (triangleVoxelCount <= 0)
    {
        data.numVoxels = 0;
        return;
    }

    data.voxels.resize(static_cast<std::size_t>(triangleVoxelCount), false);
    CUDA_LAUNCH(createTriangleVoxels, data.numTriangles, data, false);
    data.numVoxels = triangleVoxelCount;
    sortAndUnique(data.voxels, data.numVoxels);
}

void fillInterior(TetDeviceData& data, int holeCloseRadius)
{
    const int numWords = data.gridNumCells / 32 + 1;
    data.geomCells.resize(static_cast<std::size_t>(numWords), false);
    data.geomCells.setZero();
    data.airCells.resize(static_cast<std::size_t>(numWords), false);
    data.airCells.setZero();
    data.anyChanged.resize(1, false);
    data.voxelCounter.resize(1, false);

    CUDA_LAUNCH(stampGeomCells, data.numVoxels, data);

    if (holeCloseRadius > 0)
    {
        data.morphCells.resize(static_cast<std::size_t>(numWords), false);
        for (int step = 0; step < holeCloseRadius; ++step)
        {
            data.morphCells.setZero();
            CUDA_LAUNCH(dilateGeom, data.gridNumCells, data, data.geomCells.buffer, data.morphCells.buffer);
            data.geomCells.swap(data.morphCells);
        }
        for (int step = 0; step < holeCloseRadius; ++step)
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
        data.numVoxels = 0;
        return;
    }

    data.voxels.resize(static_cast<std::size_t>(solidVoxelCount), false);
    data.voxelCounter.setZero();
    CUDA_LAUNCH(collectSolidVoxels, data.gridNumCells, data, false);
    data.numVoxels = solidVoxelCount;
    sortAndUnique(data.voxels, data.numVoxels);
}

} // namespace tetrahedralizer
