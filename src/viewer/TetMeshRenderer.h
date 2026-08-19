#pragma once

#include "GlCore.h"
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
    // Per axis world space upper bound, MaxFloat where the axis is not clipped. Tets whose center is outside are
    // dropped completely, so a clipped tet is never seen from the inside.
    void setClip(const Vec3& clip);
    // In wireframe mode the tet edges are drawn, otherwise the tet faces shrunk by scale around the tet center.
    void render(bool wireframe, float scale) const;

private:
    void buildBuffers();
    void ensureShaders() const;

    std::vector<Vec3> m_nodes;
    std::vector<int> m_tet_indices;
    Vec3 m_clip{MaxFloat, MaxFloat, MaxFloat};
    GlUint m_vbo_edges = 0;
    GlUint m_vbo_faces = 0;
    int m_num_edge_vertices = 0;
    int m_num_face_vertices = 0;
    mutable GlUint m_shader_program = 0;
};

} // namespace tetrahedralizer
