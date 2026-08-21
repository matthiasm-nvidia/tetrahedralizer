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
    void uploadSizeField(const std::vector<float>& sizes);
    void setSizeFieldRange(float minSize, float maxSize);
    void render(bool wireframe = false) const;

private:
    void ensureShaders() const;
    void clearSizeField();

    mutable GlUint m_shader_program = 0;
    GlUint m_vbo_vertices = 0;
    GlUint m_vbo_sizes = 0;
    GlUint m_ibo_indices = 0;
    int m_num_indices = 0;
    int m_num_vertices = 0;
    bool m_colorize = false;
    float m_log_min = 0.0f;
    float m_log_max = 0.0f;
};

} // namespace tetrahedralizer
