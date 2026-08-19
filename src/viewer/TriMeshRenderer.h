#pragma once

#include "GlCore.h"
#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

class TriMeshRenderer
{
public:
    TriMeshRenderer() = default;
    ~TriMeshRenderer();

    TriMeshRenderer(const TriMeshRenderer&) = delete;
    TriMeshRenderer& operator=(const TriMeshRenderer&) = delete;

    void clear();
    void upload(const std::vector<Vec3>& positions, const std::vector<std::uint32_t>& triangle_indices);
    void render(bool wireframe = false) const;

private:
    void ensureShaders() const;

    mutable GlUint m_shader_program = 0;
    GlUint m_vbo_vertices = 0;
    GlUint m_ibo_indices = 0;
    int m_num_indices = 0;
};

} // namespace tetrahedralizer
