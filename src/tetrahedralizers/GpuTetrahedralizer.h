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

    void subdivide(Tetrahedralizer& mesh, float maxEdgeLength);
};

} // namespace tetrahedralizer
