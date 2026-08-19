#include "third_party/minunit.h"

#include "tetrahedralizer/Tetrahedralizer.h"
#include "tetrahedralizer/Vec.h"
#include "tetrahedralizers/TetCutTemplates.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using tetrahedralizer::Tetrahedralizer;
using tetrahedralizer::TetrahedralizerParams;
using tetrahedralizer::Vec3;

namespace
{

constexpr int kTetFaces[4][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
};

std::string faceKey(int a, int b, int c)
{
    int ids[3] = {a, b, c};
    std::sort(ids, ids + 3);
    return std::to_string(ids[0]) + "," + std::to_string(ids[1]) + "," + std::to_string(ids[2]);
}

std::pair<int, int> edgeKey(int a, int b)
{
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

float signedSixVolume(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ad = d - a;
    return ab.cross(ac).dot(ad);
}

struct MeshStats
{
    int tetCount = 0;
    int badTets = 0;
    int overusedFaces = 0;
    int boundaryFaces = 0;
    int interiorFaces = 0;
    int neighborMismatches = 0;
    int maxBoundaryFacesPerTet = 0;
    // Boundary edges used by an odd number of boundary faces. A tet edge that one
    // side splits and the other does not leaves the unsplit face unmatched, so its
    // long edge is seen by a single boundary face. Independent of node positions.
    int oddBoundaryEdges = 0;
    float volume = 0.0f;
    float minVolume = 0.0f;
};

MeshStats analyse(const Tetrahedralizer& mesh)
{
    MeshStats stats;
    stats.tetCount = mesh.numTets();
    stats.minVolume = 1.0e30f;

    std::map<std::string, int> faceCounts;
    for (int ti = 0; ti < stats.tetCount; ++ti)
    {
        const int i0 = mesh.tet_indices[4 * ti + 0];
        const int i1 = mesh.tet_indices[4 * ti + 1];
        const int i2 = mesh.tet_indices[4 * ti + 2];
        const int i3 = mesh.tet_indices[4 * ti + 3];
        const float six = signedSixVolume(mesh.nodes[i0], mesh.nodes[i1], mesh.nodes[i2], mesh.nodes[i3]);
        const float vol = six / 6.0f;
        stats.volume += vol;
        if (!(vol > 1.0e-8f))
            ++stats.badTets;
        if (vol < stats.minVolume)
            stats.minVolume = vol;

        for (const auto& face : kTetFaces)
            ++faceCounts[faceKey(mesh.tet_indices[4 * ti + face[0]], mesh.tet_indices[4 * ti + face[1]],
                                 mesh.tet_indices[4 * ti + face[2]])];
    }

    for (const auto& entry : faceCounts)
    {
        if (entry.second == 1)
            ++stats.boundaryFaces;
        else if (entry.second == 2)
            ++stats.interiorFaces;
        else
            ++stats.overusedFaces;
    }

    std::map<std::pair<int, int>, int> boundaryEdgeCounts;
    for (int ti = 0; ti < stats.tetCount; ++ti)
    {
        for (const auto& face : kTetFaces)
        {
            const int i0 = mesh.tet_indices[4 * ti + face[0]];
            const int i1 = mesh.tet_indices[4 * ti + face[1]];
            const int i2 = mesh.tet_indices[4 * ti + face[2]];
            if (faceCounts[faceKey(i0, i1, i2)] != 1)
                continue;

            ++boundaryEdgeCounts[edgeKey(i0, i1)];
            ++boundaryEdgeCounts[edgeKey(i1, i2)];
            ++boundaryEdgeCounts[edgeKey(i2, i0)];
        }
    }

    for (const auto& entry : boundaryEdgeCounts)
    {
        if (entry.second % 2 != 0)
            ++stats.oddBoundaryEdges;
    }

    if (static_cast<int>(mesh.tet_neighbors.size()) == stats.tetCount * 4)
    {
        for (int ti = 0; ti < stats.tetCount; ++ti)
        {
            int boundaryFaces = 0;
            for (int f = 0; f < 4; ++f)
            {
                const int nbr = mesh.tet_neighbors[4 * ti + f];
                if (nbr < 0)
                {
                    ++boundaryFaces;
                    continue;
                }
                if (nbr >= stats.tetCount)
                {
                    ++stats.neighborMismatches;
                    continue;
                }

                bool found = false;
                for (int g = 0; g < 4; ++g)
                {
                    if (mesh.tet_neighbors[4 * nbr + g] == ti)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    ++stats.neighborMismatches;
            }
            if (boundaryFaces > stats.maxBoundaryFacesPerTet)
                stats.maxBoundaryFacesPerTet = boundaryFaces;
        }
    }
    else
    {
        stats.neighborMismatches = stats.tetCount * 4;
    }

    return stats;
}

int countTetComponents(const Tetrahedralizer& mesh)
{
    const int tetCount = mesh.numTets();
    std::vector<int> parent(static_cast<std::size_t>(tetCount));
    for (int i = 0; i < tetCount; ++i)
        parent[static_cast<std::size_t>(i)] = i;

    const auto find = [&](auto& self, int a) -> int {
        int& p = parent[static_cast<std::size_t>(a)];
        if (p != a)
            p = self(self, p);
        return p;
    };
    const auto unite = [&](int a, int b) {
        a = find(find, a);
        b = find(find, b);
        if (a != b)
            parent[static_cast<std::size_t>(a)] = b;
    };

    for (int tet = 0; tet < tetCount; ++tet)
    {
        for (int face = 0; face < 4; ++face)
        {
            const int neighbor = mesh.tet_neighbors[4 * tet + face];
            if (neighbor >= 0)
                unite(tet, neighbor);
        }
    }

    int components = 0;
    for (int i = 0; i < tetCount; ++i)
    {
        if (find(find, i) == i)
            ++components;
    }
    return components;
}

void makeCube(std::vector<Vec3>& vertices, std::vector<std::uint32_t>& indices, float size)
{
    vertices = {
        {0, 0, 0}, {size, 0, 0}, {size, size, 0}, {0, size, 0},
        {0, 0, size}, {size, 0, size}, {size, size, size}, {0, size, size},
    };
    indices = {
        0, 2, 1, 0, 3, 2, // -Z
        4, 5, 6, 4, 6, 7, // +Z
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
    };
}

void makeUnitCube(std::vector<Vec3>& vertices, std::vector<std::uint32_t>& indices)
{
    makeCube(vertices, indices, 1.0f);
}

void makeNearbySheets(std::vector<Vec3>& vertices, std::vector<std::uint32_t>& indices)
{
    // The two parallel triangles occupy face-adjacent voxels but neither crosses
    // their shared x=1 face. The third triangle only fixes the grid origin.
    vertices = {
        {-1.0f, 3.0f, 0.2f}, {-1.0f, 3.6f, 0.2f}, {-1.0f, 3.0f, 0.8f},
        {0.75f, 0.2f, 0.2f}, {0.75f, 0.8f, 0.2f}, {0.75f, 0.2f, 0.8f},
        {1.25f, 0.2f, 0.2f}, {1.25f, 0.8f, 0.2f}, {1.25f, 0.2f, 0.8f},
    };
    indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

Tetrahedralizer makeVoxelizedCube(int cellsPerSide)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    // Axis-aligned cube of integer size with spacing 1 so flood-fill solid cells
    // line up with an N×N×N block (same role as tet-cut's 3×3×3 grid).
    makeCube(vertices, indices, static_cast<float>(cellsPerSide));

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.holeCloseRadius = 0;
    params.numOptimizationIterations = 0;
    params.projectToInputMesh = false;

    Tetrahedralizer mesh;
    mesh.create(vertices, indices, params);
    return mesh;
}

} // namespace

MU_TEST(test_cut_templates_tile_reference)
{
    tetrahedralizer::tet_cut::CutTemplateTables tables;
    tetrahedralizer::tet_cut::buildCutTemplateTables(tables);

    constexpr float kRefVol = 1.0f / 6.0f;
    int checked = 0;
    int bad = 0;

    for (int mask = 0; mask < 64; ++mask)
    {
        int ambiguous[4];
        for (int f = 0; f < 4; ++f)
            ambiguous[f] = tables.diagA[mask][f] >= 0 ? 1 : 0;

        for (int diagBits = 0; diagBits < 16; ++diagBits)
        {
            bool reachable = true;
            for (int f = 0; f < 4; ++f)
            {
                if (((diagBits >> f) & 1) && !ambiguous[f])
                    reachable = false;
            }
            if (!reachable)
                continue;

            const int childCount = tables.childCount[mask][diagBits];
            mu_check(childCount > 0);
            mu_check(childCount <= tetrahedralizer::tet_cut::kMaxChildren);

            float vol = 0.0f;
            std::map<std::string, int> faces;
            int degenerate = 0;
            for (int c = 0; c < childCount; ++c)
            {
                const int a = tables.children[mask][diagBits][c * 4 + 0];
                const int b = tables.children[mask][diagBits][c * 4 + 1];
                const int d = tables.children[mask][diagBits][c * 4 + 2];
                const int e = tables.children[mask][diagBits][c * 4 + 3];
                const float childVol = tetrahedralizer::tet_cut::refVolume(a, b, d, e);
                if (!(childVol > 1.0e-9f))
                    ++degenerate;
                vol += childVol;
                for (const auto& face : kTetFaces)
                    ++faces[faceKey(tables.children[mask][diagBits][c * 4 + face[0]],
                                    tables.children[mask][diagBits][c * 4 + face[1]],
                                    tables.children[mask][diagBits][c * 4 + face[2]])];
            }

            int overused = 0;
            for (const auto& entry : faces)
            {
                if (entry.second > 2)
                    ++overused;
            }

            ++checked;
            if (degenerate != 0 || overused != 0 || std::fabs(vol - kRefVol) > 1.0e-6f)
                ++bad;
        }
    }

    mu_assert(checked > 200, "expected hundreds of reachable templates");
    mu_assert_int_eq(0, bad);
}

MU_TEST(test_voxelized_cube_base_mesh)
{
    Tetrahedralizer mesh = makeVoxelizedCube(3);
    mu_check(!mesh.empty());
    mu_check(mesh.numTets() > 0);
    mu_check(mesh.numTets() % 5 == 0);

    const MeshStats stats = analyse(mesh);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);

    const float solidCells = static_cast<float>(mesh.numTets() / 5);
    // Each solid unit voxel becomes five tets that fill volume 1.
    mu_check(std::fabs(stats.volume - solidCells) < 0.05f);
    // Discrete solid should cover at least the input cube volume.
    mu_check(stats.volume + 0.05f >= 27.0f);
}

MU_TEST(test_split_voxels_keeps_interior_connected)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 3.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.numOptimizationIterations = 0;
    params.projectToInputMesh = false;

    Tetrahedralizer mesh;
    mesh.create(vertices, indices, params);

    const int voxelCount = mesh.numTets() / 5;
    // Interior voxels share nodes; a failed merge leaves 8 corners per voxel.
    mu_check(static_cast<int>(mesh.nodes.size()) < voxelCount * 8);
    mu_check(static_cast<int>(mesh.nodes.size()) < voxelCount * 4);

    const MeshStats stats = analyse(mesh);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_check(std::fabs(stats.volume - static_cast<float>(voxelCount)) < 0.05f);
}

MU_TEST(test_split_voxels_disconnects_nearby_sheets)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeNearbySheets(vertices, indices);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.numOptimizationIterations = 0;
    params.projectToInputMesh = false;

    Tetrahedralizer mesh;
    mesh.create(vertices, indices, params);

    mu_check(!mesh.empty());
    // Origin triangle plus two face-adjacent sheets that must not glue together.
    mu_check(countTetComponents(mesh) >= 3);

    const MeshStats stats = analyse(mesh);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
}

MU_TEST(test_subdivision_preserves_manifold_volume)
{
    Tetrahedralizer mesh = makeVoxelizedCube(3);
    const MeshStats before = analyse(mesh);
    mu_assert_int_eq(0, before.badTets);

    const std::size_t nodeCount = mesh.nodes.size();
    const int tetCount = mesh.numTets();
    mesh.subdivide(1.1f);

    const MeshStats stats = analyse(mesh);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_check(mesh.nodes.size() > nodeCount);
    mu_check(mesh.numTets() > tetCount);
    mu_check(std::fabs(stats.volume - before.volume) < 5.0e-3f);
}

MU_TEST(test_max_edge_length_parameter)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 3.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.numOptimizationIterations = 0;
    params.projectToInputMesh = false;

    Tetrahedralizer baseMesh;
    baseMesh.create(vertices, indices, params);

    params.maxEdgeLength = 1.1f;
    Tetrahedralizer subdivided;
    subdivided.create(vertices, indices, params);
    mu_check(!subdivided.empty());

    const MeshStats stats = analyse(subdivided);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_check(subdivided.nodes.size() > baseMesh.nodes.size());
    mu_check(subdivided.numTets() > baseMesh.numTets());
    mu_check(std::fabs(stats.volume - analyse(baseMesh).volume) < 5.0e-3f);
}

MU_TEST(test_project_to_input_mesh_smoke)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 3.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.numOptimizationIterations = 0;
    params.projectToInputMesh = false;

    Tetrahedralizer baseMesh;
    baseMesh.create(vertices, indices, params);

    params.projectToInputMesh = true;
    Tetrahedralizer projected;
    projected.create(vertices, indices, params);
    mu_check(!projected.empty());
    mu_check(projected.numTets() > baseMesh.numTets());
    mu_check(projected.nodes.size() > baseMesh.nodes.size());

    int moved = 0;
    for (std::size_t i = 0; i < baseMesh.nodes.size(); ++i)
    {
        if ((projected.nodes[i] - baseMesh.nodes[i]).magnitudeSquared() > 1.0e-12f)
            ++moved;
    }
    mu_check(moved > 0);

    const MeshStats stats = analyse(projected);
    mu_assert_int_eq(0, stats.badTets);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_assert_int_eq(0, stats.oddBoundaryEdges);
    mu_check(stats.maxBoundaryFacesPerTet <= 1);
}

MU_TEST(test_boundary_refinement_stays_conforming)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 4.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.projectToInputMesh = true;
    params.numOptimizationIterations = 0;

    Tetrahedralizer refined;
    refined.create(vertices, indices, params);
    mu_check(!refined.empty());

    const MeshStats refinedStats = analyse(refined);
    mu_assert_int_eq(0, refinedStats.badTets);
    mu_assert_int_eq(0, refinedStats.overusedFaces);
    mu_assert_int_eq(0, refinedStats.neighborMismatches);
    mu_assert_int_eq(0, refinedStats.oddBoundaryEdges);
    mu_check(refinedStats.maxBoundaryFacesPerTet <= 1);

    params.numOptimizationIterations = 10;
    Tetrahedralizer optimized;
    optimized.create(vertices, indices, params);

    // Smoothing only moves nodes, so the refined connectivity must survive it, and
    // the step backoff must keep every tet from inverting.
    const MeshStats optimizedStats = analyse(optimized);
    mu_assert_int_eq(refined.numTets(), optimized.numTets());
    mu_assert_int_eq(0, optimizedStats.overusedFaces);
    mu_assert_int_eq(0, optimizedStats.neighborMismatches);
    mu_assert_int_eq(0, optimizedStats.oddBoundaryEdges);
    mu_assert_int_eq(0, optimizedStats.badTets);
    mu_check(optimizedStats.minVolume > 0.0f);
}

MU_TEST(test_optimization_loop_smoke)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 3.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.projectToInputMesh = true;
    params.numOptimizationIterations = 5;
    params.volumeFactor = 0.8f;
    params.edgeContraction = 0.5f;

    Tetrahedralizer mesh;
    mesh.create(vertices, indices, params);
    mu_check(!mesh.empty());

    const MeshStats stats = analyse(mesh);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_assert_int_eq(0, stats.oddBoundaryEdges);
    mu_assert_int_eq(0, stats.badTets);
    mu_check(stats.boundaryFaces > 0);
    mu_check(stats.maxBoundaryFacesPerTet <= 1);
}

MU_TEST(test_smoothing_without_projection_moves_nodes)
{
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;
    makeCube(vertices, indices, 3.0f);

    TetrahedralizerParams params;
    params.voxelSpacing = 1.0f;
    params.projectToInputMesh = false;
    params.numOptimizationIterations = 0;
    params.volumeFactor = 0.8f;

    Tetrahedralizer baseMesh;
    baseMesh.create(vertices, indices, params);

    params.numOptimizationIterations = 5;
    params.edgeContraction = 0.2f;
    Tetrahedralizer smoothed;
    smoothed.create(vertices, indices, params);
    mu_check(!smoothed.empty());
    mu_assert_int_eq(baseMesh.numTets(), smoothed.numTets());
    mu_assert_int_eq(static_cast<int>(baseMesh.nodes.size()), static_cast<int>(smoothed.nodes.size()));

    int moved = 0;
    for (std::size_t i = 0; i < baseMesh.nodes.size(); ++i)
    {
        if ((smoothed.nodes[i] - baseMesh.nodes[i]).magnitudeSquared() > 1.0e-12f)
            ++moved;
    }
    mu_check(moved > 0);

    const MeshStats stats = analyse(smoothed);
    mu_assert_int_eq(0, stats.overusedFaces);
    mu_assert_int_eq(0, stats.neighborMismatches);
    mu_assert_int_eq(0, stats.badTets);
}

MU_TEST_SUITE(test_suite)
{
    MU_RUN_TEST(test_cut_templates_tile_reference);
    MU_RUN_TEST(test_voxelized_cube_base_mesh);
    MU_RUN_TEST(test_split_voxels_keeps_interior_connected);
    MU_RUN_TEST(test_split_voxels_disconnects_nearby_sheets);
    MU_RUN_TEST(test_subdivision_preserves_manifold_volume);
    MU_RUN_TEST(test_max_edge_length_parameter);
    MU_RUN_TEST(test_project_to_input_mesh_smoke);
    MU_RUN_TEST(test_boundary_refinement_stays_conforming);
    MU_RUN_TEST(test_optimization_loop_smoke);
    MU_RUN_TEST(test_smoothing_without_projection_moves_nodes);
}

int main()
{
    MU_RUN_SUITE(test_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
