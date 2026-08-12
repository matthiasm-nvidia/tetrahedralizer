#pragma once

#include "tetrahedralizer/GlCore.h"
#include "tetrahedralizer/Vec.h"

#include <vector>

namespace tetrahedralizer
{

class TetMeshRenderer
{
public:
    TetMeshRenderer() = default;
    ~TetMeshRenderer();

    TetMeshRenderer(const TetMeshRenderer&) = delete;
    TetMeshRenderer& operator=(const TetMeshRenderer&) = delete;

    void clear();
    void upload(const std::vector<Vec3>& nodes, const std::vector<int>& tet_indices);
    void render() const;

private:
    GlUint m_vbo_vertices = 0;
    int m_num_vertices = 0;
};

} // namespace tetrahedralizer
