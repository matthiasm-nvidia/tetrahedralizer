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
    // When true, find one cut per tet edge against the input surface mesh.
    bool cutWithInputMesh = false;
    // Shape-matching iterations that drive each tet toward a regular tet.
    // 0 skips smoothing.
    int numSmoothingIterations = 0;
    // Target regular-tet volume as a fraction of the current tet volume (< 1 contracts).
    float volumeFactor = 0.8f;
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

    // Cut unique tet edges at midpoints with the given probability, then split (GPU).
    void cutRandomEdges(float probability, unsigned seed = 1);
    // Apply a 6-bit edge-cut mask to a single-tet mesh, then split (GPU).
    void cutSingleTetByMask(int mask);

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
