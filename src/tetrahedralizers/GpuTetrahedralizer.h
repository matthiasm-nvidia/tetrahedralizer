#pragma once

#include "tetrahedralizer/Tetrahedralizer.h"
#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

class GpuTetrahedralizer
{
public:
    // Host-side fallback used until a GPU path is wired into the build.
    void create(Tetrahedralizer& output, const std::vector<Vec3>& mesh_vertices,
                const std::vector<std::uint32_t>& mesh_indices);
};

} // namespace tetrahedralizer
