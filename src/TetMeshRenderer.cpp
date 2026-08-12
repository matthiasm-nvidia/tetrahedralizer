#include "tetrahedralizer/TetMeshRenderer.h"

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

namespace tetrahedralizer
{
namespace
{

constexpr int kTetEdgePairs[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},
};

} // namespace

TetMeshRenderer::~TetMeshRenderer()
{
    clear();
}

void TetMeshRenderer::clear()
{
    if (m_vbo_vertices != 0)
    {
        glDeleteBuffers(1, &m_vbo_vertices);
        m_vbo_vertices = 0;
    }
    m_num_vertices = 0;
}

void TetMeshRenderer::upload(const std::vector<Vec3>& nodes, const std::vector<int>& tet_indices)
{
    clear();

    if (nodes.empty() || tet_indices.size() < 4)
        return;

    const int num_tets = static_cast<int>(tet_indices.size() / 4);
    std::vector<Vec3> segments;
    segments.reserve(static_cast<std::size_t>(num_tets) * 12);

    for (int t = 0; t < num_tets; ++t)
    {
        const int base = t * 4;
        int corners[4];
        bool valid = true;
        for (int k = 0; k < 4; ++k)
        {
            corners[k] = tet_indices[static_cast<std::size_t>(base + k)];
            if (corners[k] < 0 || corners[k] >= static_cast<int>(nodes.size()))
            {
                valid = false;
                break;
            }
        }
        if (!valid)
            continue;

        for (const auto& edge : kTetEdgePairs)
        {
            segments.push_back(nodes[static_cast<std::size_t>(corners[edge[0]])]);
            segments.push_back(nodes[static_cast<std::size_t>(corners[edge[1]])]);
        }
    }

    if (segments.empty())
        return;

    m_num_vertices = static_cast<int>(segments.size());
    glGenBuffers(1, &m_vbo_vertices);
    glBindBuffer(kArrayBuffer, m_vbo_vertices);
    glBufferData(kArrayBuffer, static_cast<GlSize>(segments.size() * sizeof(Vec3)), segments.data(), kStaticDraw);
    glBindBuffer(kArrayBuffer, 0);
}

void TetMeshRenderer::render() const
{
    if (m_num_vertices <= 0 || m_vbo_vertices == 0)
        return;

    glDisableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(kArrayBuffer, m_vbo_vertices);
    glVertexPointer(3, kFloat, 0, nullptr);

    glColor3f(0.95f, 0.75f, 0.20f);
    glDrawArrays(kLines, 0, m_num_vertices);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(kArrayBuffer, 0);
}

} // namespace tetrahedralizer
