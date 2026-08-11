#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

class TriMesh
{
public:
    bool loadObj(const char* path);
    Bounds3 bounds() const;

    bool empty() const
    {
        return positions.empty() || triangle_indices.empty();
    }

    std::vector<Vec3> positions;
    std::vector<std::uint32_t> triangle_indices;
};

} // namespace tetrahedralizer
