#include "GpuTetrahedralizer.h"

#include "utils/CudaUtils.h"

#include <vector>

namespace tetrahedralizer
{

struct TetDeviceData
{
    int numNodes = 0;
    int numTets = 0;

    DeviceBuffer<Vec3> nodes;
    DeviceBuffer<int> tet_indices;

    void free()
    {
        nodes.free();
        tet_indices.free();
        numNodes = 0;
        numTets = 0;
    }
};

void GpuTetrahedralizer::create(Tetrahedralizer& output, const std::vector<Vec3>& mesh_vertices,
                                const std::vector<std::uint32_t>& mesh_indices)
{
    // Simple construction currently runs on the host; device buffers are filled for later GPU work.
    output.create(mesh_vertices, mesh_indices);

    TetDeviceData data;
    data.numNodes = static_cast<int>(output.nodes.size());
    data.numTets = output.numTets();
    if (data.numNodes > 0)
        data.nodes.set(output.nodes);
    if (!output.tet_indices.empty())
        data.tet_indices.set(output.tet_indices);
    data.free();
}

} // namespace tetrahedralizer
