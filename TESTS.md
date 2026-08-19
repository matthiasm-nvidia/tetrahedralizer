# Tests

GPU / host regression checks for tetrahedralization, edge subdivision, and face neighbors.
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

- Tet count is a positive multiple of 5
- All tet volumes \(> 0\)
- Face conformity (each face used once or twice)
- `tet_neighbors` mutual consistency
- Total tet volume ≈ number of solid voxels (unit spacing), and at least the input cube volume

### `test_split_voxels_keeps_interior_connected`

Voxelize a solid cube with `splitVoxels`. Interior voxels must still share nodes (node count well below `8 × voxelCount`); volume and manifold checks match the unsplit mesh.

### `test_split_voxels_disconnects_nearby_sheets`

Voxelize two nearby parallel triangles in face-adjacent cells. With `splitVoxels` enabled:

- Tet count and total volume stay unchanged
- Additional node IDs keep the sheets topologically disconnected
- Tet volumes, face usage, and neighbors remain valid

### `test_subdivision_preserves_manifold_volume`

Start from the 3×3×3 voxel mesh and call `subdivide(1.1)`. Checks:

- Manifold faces / valid neighbors
- Positive volumes
- Total volume preserved (float tolerance)
- Node and tet counts increase

### `test_max_edge_length_parameter`

Run the full create pipeline with `maxEdgeLength = 1.1`. Checks manifold / positive volumes, preserved volume, and increased node and tet counts relative to the unsubdivided mesh.

### `test_project_to_input_mesh_smoke`

Voxelize a cube with `projectToInputMesh`. Boundary refinement increases the tet/node counts, preserves a valid manifold mesh, and leaves every tet with at most one boundary face before projection.

### `test_boundary_refinement_stays_conforming`

Side-4 cube with `projectToInputMesh`, once without and once with optimization iterations. Checks that boundary refinement leaves a conforming mesh (positive volumes, face usage, neighbors, at most one boundary face per tet), that the connectivity survives the optimization loop, and that all volumes stay positive afterwards (the step backoff in `applyNodeMovesSafely`).

Conformity uses `oddBoundaryEdges`: every boundary edge must be used by an even number of boundary faces. If one tet splits an edge and its neighbor does not, the unsplit face stays unmatched and its long edge is seen by a single boundary face. The count is purely topological, so it still detects such a T-junction after nodes have moved, unlike a colinearity test.

### `test_optimization_loop_smoke`

Voxelize a cube with `projectToInputMesh` and `numOptimizationIterations = 5`. Checks the mesh stays manifold with boundary faces, with at most one boundary face per tet, after smooth→project loops.

## Adding tests

Prefer checks on real invariants (volume, manifold, neighbors, template tiling). Skip trivial getters. New GPU behavior that changes connectivity should extend the subdivision suites rather than only adding viewer-side checks.
