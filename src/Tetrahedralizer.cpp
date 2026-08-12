#include "tetrahedralizer/Tetrahedralizer.h"

namespace tetrahedralizer
{

void Tetrahedralizer::clear()
{
    nodes.clear();
    tet_indices.clear();
}

void Tetrahedralizer::create(const std::vector<Vec3>& mesh_vertices,
                             const std::vector<std::uint32_t>& mesh_indices)
{
    clear();

    if (mesh_vertices.empty() || mesh_indices.size() < 3)
        return;

    Bounds3 bounds(Empty);
    for (const Vec3& p : mesh_vertices)
        bounds.include(p);
    const Vec3 center = bounds.getCenter();

    nodes.reserve(mesh_vertices.size() + 1);
    nodes.push_back(center);
    nodes.insert(nodes.end(), mesh_vertices.begin(), mesh_vertices.end());

    const int num_tris = static_cast<int>(mesh_indices.size() / 3);
    tet_indices.reserve(static_cast<std::size_t>(num_tris) * 4);
    for (int t = 0; t < num_tris; ++t)
    {
        const int a = static_cast<int>(mesh_indices[static_cast<std::size_t>(t) * 3 + 0]) + 1;
        const int b = static_cast<int>(mesh_indices[static_cast<std::size_t>(t) * 3 + 1]) + 1;
        const int c = static_cast<int>(mesh_indices[static_cast<std::size_t>(t) * 3 + 2]) + 1;
        tet_indices.push_back(0);
        tet_indices.push_back(a);
        tet_indices.push_back(b);
        tet_indices.push_back(c);
    }
}

} // namespace tetrahedralizer
