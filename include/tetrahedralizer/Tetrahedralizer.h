#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

struct TetrahedralizerParams
{
    float voxelSpacing = 1.0f;
};

class Tetrahedralizer
{
public:
    Tetrahedralizer() = default;

    std::vector<Vec3> nodes;
    std::vector<int> tet_indices;

    void clear();
    void create(const std::vector<Vec3>& mesh_vertices, const std::vector<std::uint32_t>& mesh_indices,
                const TetrahedralizerParams& params = {});

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
