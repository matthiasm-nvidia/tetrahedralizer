#include "tetrahedralizer/Tetrahedralizer.h"

#ifdef TETRAHEDRALIZER_HAS_CUDA
#include "tetrahedralizers/GpuTetrahedralizer.h"
#endif

#include <stdexcept>

namespace tetrahedralizer
{

namespace
{

constexpr int kTetFaces[4][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
};

} // namespace

void Tetrahedralizer::clear()
{
    nodes.clear();
    tet_indices.clear();
    tet_neighbors.clear();
}

void Tetrahedralizer::create(const std::vector<Vec3>& mesh_vertices,
                             const std::vector<std::uint32_t>& mesh_indices,
                             const TetrahedralizerParams& params)
{
    clear();

    if (mesh_vertices.empty() || mesh_indices.size() < 3)
        return;

#ifdef TETRAHEDRALIZER_HAS_CUDA
    GpuTetrahedralizer{}.create(*this, mesh_vertices, mesh_indices, params);
#else
    (void)params;
    throw std::runtime_error("Tetrahedralizer was built without CUDA support");
#endif
}

void Tetrahedralizer::subdivide(float maxEdgeLength)
{
#ifdef TETRAHEDRALIZER_HAS_CUDA
    GpuTetrahedralizer{}.subdivide(*this, maxEdgeLength);
#else
    (void)maxEdgeLength;
    throw std::runtime_error("Tetrahedralizer was built without CUDA support");
#endif
}

std::vector<Vec3> Tetrahedralizer::nodeNormals() const
{
    std::vector<Vec3> normals(nodes.size(), Vec3(Zero));
    const int tetCount = numTets();
    if (tetCount <= 0 || tet_neighbors.size() != tet_indices.size())
        return normals;

    for (int tetIndex = 0; tetIndex < tetCount; ++tetIndex)
    {
        const int* ids = tet_indices.data() + 4 * tetIndex;
        for (int face = 0; face < 4; ++face)
        {
            if (tet_neighbors[static_cast<std::size_t>(4 * tetIndex + face)] >= 0)
                continue;

            const int i0 = ids[kTetFaces[face][0]];
            const int i1 = ids[kTetFaces[face][1]];
            const int i2 = ids[kTetFaces[face][2]];
            const int iOpp = ids[6 - kTetFaces[face][0] - kTetFaces[face][1] - kTetFaces[face][2]];
            if (i0 < 0 || i1 < 0 || i2 < 0 || iOpp < 0 ||
                i0 >= static_cast<int>(nodes.size()) || i1 >= static_cast<int>(nodes.size()) ||
                i2 >= static_cast<int>(nodes.size()) || iOpp >= static_cast<int>(nodes.size()))
                continue;

            const Vec3 p0 = nodes[static_cast<std::size_t>(i0)];
            Vec3 normal = (nodes[static_cast<std::size_t>(i1)] - p0).cross(nodes[static_cast<std::size_t>(i2)] - p0);
            if (normal.dot(nodes[static_cast<std::size_t>(iOpp)] - p0) > 0.0f)
                normal = -normal;

            normals[static_cast<std::size_t>(i0)] += normal;
            normals[static_cast<std::size_t>(i1)] += normal;
            normals[static_cast<std::size_t>(i2)] += normal;
        }
    }

    for (Vec3& normal : normals)
    {
        if (normal.magnitudeSquared() > 0.0f)
            normal.normalize();
    }
    return normals;
}

} // namespace tetrahedralizer
