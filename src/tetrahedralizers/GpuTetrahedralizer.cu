#include "GpuTetrahedralizer.h"
#include "TetDeviceData.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tetrahedralizer
{

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

void GpuTetrahedralizer::create(Tetrahedralizer& output, const std::vector<Vec3>& mesh_vertices,
                                const std::vector<std::uint32_t>& mesh_indices,
                                const TetrahedralizerParams& params)
{
    if (!(params.voxelSpacing > 0.0f) || !std::isfinite(params.voxelSpacing))
        throw std::invalid_argument("Voxel spacing must be finite and greater than zero");
    if (params.holeCloseRadius < 0)
        throw std::invalid_argument("Hole close radius must be non-negative");
    if (params.numOptimizationIterations < 0)
        throw std::invalid_argument("Optimization iteration count must be non-negative");
    if (!(params.volumeContraction >= 0.0f) || !std::isfinite(params.volumeContraction))
        throw std::invalid_argument("Volume contraction must be finite and non-negative");
    if (!(params.edgeContraction >= 0.0f) || !std::isfinite(params.edgeContraction))
        throw std::invalid_argument("Edge contraction must be finite and non-negative");
    if (!(params.maxEdgeLength >= 0.0f) || !std::isfinite(params.maxEdgeLength))
        throw std::invalid_argument("Maximum edge length must be finite and non-negative");
    if (!(params.minEdgeLength >= 0.0f) || !std::isfinite(params.minEdgeLength))
        throw std::invalid_argument("Minimum edge length must be finite and non-negative");
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

        voxelizeTriangles(data);
        if (data.numVoxels <= 0)
        {
            data.free();
            return;
        }

        fillInterior(data, params.holeCloseRadius);
        if (data.numVoxels <= 0)
        {
            data.free();
            return;
        }

        buildInputMeshBvh(data);

        if (data.numVoxels > std::numeric_limits<int>::max() / 8)
            throw std::runtime_error("Voxel corner count exceeds the supported range");

        createSplitVoxelNodes(data);
        createFiveTets(data);

        if (params.maxEdgeLength > 0.0f)
            subdivideLongEdges(data, params.maxEdgeLength);
        if (params.minEdgeLength > 0.0f)
            collapseShortEdges(data, params.minEdgeLength);

        cudaCheck(cudaDeviceSynchronize());

        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after tetrahedralization");

        separateBoundaryFaces(data);
        runOptimization(data, params);

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

void GpuTetrahedralizer::subdivide(Tetrahedralizer& mesh, float maxEdgeLength)
{
    if (!(maxEdgeLength >= 0.0f) || !std::isfinite(maxEdgeLength))
        throw std::invalid_argument("Maximum edge length must be finite and non-negative");
    if (maxEdgeLength == 0.0f)
        return;

    TetDeviceData data;
    try
    {
        uploadMeshFromHost(data, mesh);
        subdivideLongEdges(data, maxEdgeLength);

        cudaCheck(cudaDeviceSynchronize());
        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after subdivision");

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
