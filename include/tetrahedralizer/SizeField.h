#pragma once

#include "tetrahedralizer/Vec.h"

#include <cstdint>
#include <vector>

namespace tetrahedralizer
{

struct SizeFieldParams
{
    // Max distance from a surface chord to the input mesh (ε).
    float geometricError = 0.01f;
    // Floor for h_max. 0 skips the extra floor.
    float minSize = 0.0f;
    // Ceiling for h_max. 0 uses the input-mesh bounding diagonal.
    float maxSize = 0.0f;
    // Jacobi-style averaging of h_max over incident triangles. 0 skips.
    int smoothingIterations = 3;
};

// One h_max per input vertex: local maximum surface edge length from max turning.
std::vector<float> computeSurfaceSizeField(const std::vector<Vec3>& positions,
                                           const std::vector<std::uint32_t>& triangle_indices,
                                           const SizeFieldParams& params = {});

} // namespace tetrahedralizer
