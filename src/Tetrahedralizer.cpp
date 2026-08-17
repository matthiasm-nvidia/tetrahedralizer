#include "tetrahedralizer/Tetrahedralizer.h"

#ifdef TETRAHEDRALIZER_HAS_CUDA
#include "tetrahedralizers/GpuTetrahedralizer.h"
#endif

#include <stdexcept>

namespace tetrahedralizer
{

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

void Tetrahedralizer::cutRandomEdges(float probability, unsigned seed)
{
#ifdef TETRAHEDRALIZER_HAS_CUDA
    GpuTetrahedralizer{}.cutRandomEdges(*this, probability, seed);
#else
    (void)probability;
    (void)seed;
    throw std::runtime_error("Tetrahedralizer was built without CUDA support");
#endif
}

void Tetrahedralizer::cutSingleTetByMask(int mask)
{
#ifdef TETRAHEDRALIZER_HAS_CUDA
    GpuTetrahedralizer{}.cutSingleTetByMask(*this, mask);
#else
    (void)mask;
    throw std::runtime_error("Tetrahedralizer was built without CUDA support");
#endif
}

} // namespace tetrahedralizer
