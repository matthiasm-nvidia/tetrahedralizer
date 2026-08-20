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
    // Collapse tet edges shorter than this value. 0 skips collapse.
    float minEdgeLength = 0.0f;
    // Project boundary nodes onto the input mesh along estimated outward normals.
    bool projectToInputMesh = true;
    // When projecting, snap each boundary node to the closest input-mesh point
    // instead of raycasting along its estimated normal. No offset; moves inward or outward.
    bool projectToClosestPoint = false;
    // Optimization iterations: each does tet smoothing, then edge smoothing, then
    // project (if enabled). When projecting, surface normals are recomputed before
    // each smooth so surface nodes only slide tangentially, and again before project.
    // 0 skips smoothing; projection still runs once if enabled.
    int numOptimizationIterations = 15;
    // Fraction of current tet volume to remove while shape-matching to a regular tet.
    // 0 skips tet smoothing; 0.2 targets 80% of the current volume.
    float volumeContraction = 0.2f;
    // Pull opposite endpoints of each tet edge toward each other. 0 skips edge smoothing.
    float edgeContraction = 0.2f;
    // Edge smoothing only. On: every edge contracts both ends, then surface nodes keep
    // only the tangential part of the correction. Off: mixed interior/surface edges move
    // only the interior node, with no tangent strip.
    bool useNormals = false;
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

    // One outward unit normal per node from boundary faces; interior nodes are zero.
    std::vector<Vec3> nodeNormals() const;
};

} // namespace tetrahedralizer
