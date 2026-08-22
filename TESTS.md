# Tests

GPU / host regression checks for tetrahedralization, edge subdivision, adaptive remesh, and face neighbors.
Run with:

```bat
test.bat
```

This configures (if needed), builds `tetrahedralizer_tests` (Release), and runs it. Use the same Visual Studio / CUDA environment as `build.bat`.

## Executable

| Target | Sources |
| --- | --- |
| `tetrahedralizer_tests` | `tests/test_tetrahedralizer.cpp` + `Tetrahedralizer` / CUDA core (no viewer) |

Framework: `include/third_party/minunit.h`.

## Fixtures

- **Cube mesh** — axis-aligned cube of side \(N\) with `voxelSpacing = 1` (same role as tet-cut’s block). Exact solid-cell count depends on surface stamping; tests assert volume ≈ \(N^3\) and a 5-tet decomposition rather than a fixed tet count.

## Suites

### `test_cut_templates_tile_reference`

Host-only. Builds the 64×diagBits cut tables and checks every reachable `(mask, diagBits)`:

- Positive child volumes on the reference tet
- Children tile volume \(1/6\)
- No face shared by more than two child tets

### `test_voxelized_cube_base_mesh`

Voxelize a side-3 cube at spacing 1:

- Tet count is positive
- All tet volumes \(> 0\)
- Face conformity (each face used once or twice)
- `tet_neighbors` mutual consistency
- Total tet volume at least the input cube volume (empty exterior tets of surface voxels are removed)

### `test_split_voxels_keeps_interior_connected`

Voxelize a solid cube. Interior voxels must still share nodes (node count well below 8 per 5 tets); volume and manifold checks remain valid.

### `test_split_voxels_disconnects_nearby_sheets`

Voxelize two nearby parallel triangles in face-adjacent cells. Additional node IDs keep the sheets topologically disconnected; tet volumes, face usage, and neighbors remain valid.

### `test_empty_exterior_tets_removed`

Voxelize a single open triangle (no enclosed volume). With `removeEmptyExteriorTets` on (default), empty exterior tets are deleted so every remaining tet intersects the input triangle. Off keeps extra empty tets.

### `test_subdivision_preserves_manifold_volume`

Start from the 3×3×3 voxel mesh and call `subdivide(1.1)`. Checks:

- Manifold faces / valid neighbors
- Positive volumes
- Total volume preserved (float tolerance)
- Node and tet counts increase

### `test_adaptive_off_ignores_edge_scales`

Cube create with tight min/max and a small ε, but `numAdaptiveIterations = 0`. Node and tet counts match the default (non-adaptive) run.

### `test_max_edge_length_parameter`

Adaptive create with `numAdaptiveIterations = 1` and `maxEdgeLength = 1.1` (voxel units) on a cube (smooth contractions off). Checks manifold / positive volumes and increased node and tet counts relative to the unsubdivided mesh.

### `test_project_to_input_mesh_smoke`

Voxelize a cube with `projectToInputMesh` (normal raycast) and one optimization iteration (smooth contractions off). Some boundary nodes move, tet/node counts match the unprojected mesh, and the mesh stays manifold with at most one boundary face per tet.

### `test_project_to_closest_point_smoke`

Same cube with `projectToClosestPoint`. Some boundary nodes move; manifold and boundary-face checks match the raycast smoke test.

### `test_boundary_refinement_stays_conforming`

Side-4 cube with `projectToInputMesh`, once without and once with optimization iterations. Checks that boundary refinement leaves a conforming mesh (positive volumes, face usage, neighbors, at most one boundary face per tet), that the connectivity survives the optimization loop, and that all volumes stay positive afterwards (the step backoff in `applyNodeMovesSafely`).

Conformity uses `oddBoundaryEdges`: every boundary edge must be used by an even number of boundary faces. If one tet splits an edge and its neighbor does not, the unsplit face stays unmatched and its long edge is seen by a single boundary face. The count is purely topological, so it still detects such a T-junction after nodes have moved, unlike a colinearity test.

### `test_optimization_loop_smoke`

Voxelize a cube with `projectToInputMesh` and `numOptimizationIterations = 5`. Checks the mesh stays manifold with boundary faces, with at most one boundary face per tet, after smooth→project loops.

### `test_node_normals_mark_boundary`

Voxelized cube. `nodeNormals()` is unit-length and outward on boundary nodes, zero on interior nodes.

### `test_smoothing_without_projection_moves_nodes`

Cube with five optimization iterations, projection off. Node and tet counts stay the same; some nodes move; the mesh stays manifold.

### `test_dragon_voxel_spacing`

Load `tests/data/dragon.obj`, voxelize at spacing `0.1` with optimize/project off. Mesh is non-empty, manifold, positive volumes, at most one boundary face per tet.

### `test_dragon_adaptive_stays_manifold`

Same dragon at spacing `0.1` with `numAdaptiveIterations = 1` (opt/project off). Stays manifold after one size-field split pass.

### `test_dragon_adaptive_fine_stays_manifold`

Same as above at spacing `0.05`.

### `test_adaptive_size_field_cube_stays_manifold`

Cube create with `numAdaptiveIterations = 1` and a small ε. Stays manifold with at most one boundary face per tet.

### `test_adaptive_size_field_refines_curved`

Octahedron at spacing `0.25`. Adaptive create with a small ε produces more tets than the non-adaptive run and stays manifold.

### `test_size_field_flat_is_coarse`

Host-only. A flat quad's size field is at the global ceiling (no turning).

### `test_size_field_turning_refines`

Host-only. A bent pair of triangles gets a smaller `h_max` at the crease than a flat quad.

### `test_cpu_bvh_raycast_hits_triangle`

Host-only. CPU BVH raycast hits a known triangle at the expected `t` and misses a ray that does not intersect.

### `test_triangle_tetrahedron_intersection`

Host-only. SAT test hits a tet when a triangle vertex is inside, when a plane stabs through, and when a tet face is coplanar; misses a disjoint triangle.

## Adding tests

Prefer checks on real invariants (volume, manifold, neighbors, template tiling). Skip trivial getters. New GPU behavior that changes connectivity should extend the subdivision suites rather than only adding viewer-side checks.
