#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

class Tetrahedralizer;
struct TetrahedralizerParams;

class GpuTetrahedralizer
{
public:
    void create(Tetrahedralizer& output, const std::vector<Vec3>& mesh_vertices,
                const std::vector<std::uint32_t>& mesh_indices, const TetrahedralizerParams& params);

    // Test helpers: cut existing tet edges at midpoints, then run Steiner + template split.
    void cutRandomEdges(Tetrahedralizer& mesh, float probability, unsigned seed);
    // Requires mesh.numTets() == 1. Mask bits match tet-cut edge order.
    void cutSingleTetByMask(Tetrahedralizer& mesh, int mask);
};

} // namespace tetrahedralizer
