#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

struct TetrahedralizerParams
{
    float voxelSpacing = 1.0f;
    // Morphological close radius in voxels before the exterior flood fill.
    // 0 skips closing; larger values seal bigger shell holes.
    int holeCloseRadius = 0;
    // Subdivide tet edges longer than this value at their midpoints.
    // 0 skips subdivision.
    float maxEdgeLength = 0.0f;
    // Project boundary nodes onto the input mesh along estimated outward normals.
    bool projectToInputMesh = true;
    // Optimization iterations: each does project (if enabled), tet smoothing, then edge
    // smoothing. When projecting, surface normals are recomputed before projection and
    // before each smooth so surface nodes only slide tangentially. 0 skips.
    int numOptimizationIterations = 15;
    // Target regular-tet volume as a fraction of the current tet volume (< 1 contracts).
    // 0 skips tet smoothing.
    float volumeFactor = 0.8f;
    // Pull opposite endpoints of each tet edge toward each other. 0 skips edge smoothing.
    float edgeContraction = 0.0f;
    // Edge smoothing only. On: every edge contracts both ends, then surface nodes keep
    // only the tangential part of the correction. Off: mixed interior/surface edges move
    // only the interior node, with no tangent strip.
    bool useNormals = true;
};

class Tetrahedralizer
{
public:
    Tetrahedralizer() = default;

    std::vector<Vec3> nodes;
    std::vector<int> tet_indices;
    // 4 entries per tet: adjacent tet index for each local face, or -1 on the boundary.
    std::vector<int> tet_neighbors;

    void clear();
    void create(const std::vector<Vec3>& mesh_vertices, const std::vector<std::uint32_t>& mesh_indices,
                const TetrahedralizerParams& params = {});

    // Subdivide unique tet edges longer than maxEdgeLength at their midpoints (GPU).
    // 0 leaves the mesh unchanged.
    void subdivide(float maxEdgeLength);

    bool empty() const
    {
        return nodes.empty() || tet_indices.empty();
    }

    int numTets() const
    {
        return static_cast<int>(tet_indices.size() / 4);
    }
};

} // namespace tetrahedralizer
