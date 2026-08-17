# Tests

GPU / host regression checks for tetrahedralization, edge cutting, and face neighbors.
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
- **Reference tet** — corners `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)` for exhaustive mask tests.
- **Forced edge cuts** — `Tetrahedralizer::cutRandomEdges` / `cutSingleTetByMask` mark unique tet edges at midpoints, then run the same Steiner + template split path as surface cutting (no BVH). Needed so random cuts exercise interior edges, not only surface hits.

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

### `test_random_cuts_preserve_manifold_volume`

Start from the 3×3×3 voxel mesh, then three rounds of `cutRandomEdges` (probabilities 0.35, 0.15, 0.15). After each round:

- Manifold faces / valid neighbors
- Positive volumes
- Total volume preserved (float tolerance)

### `test_all_masks_on_single_tet`

For each mask in \(0..63\), cut the reference tet with that edge mask and check volume \(1/6\), conformity, and neighbors.

### `test_surface_cut_smoke`

Full pipeline: voxelize the cube with `cutWithInputMesh`, cutting against an extra plane through the cube. Checks manifold / positive volumes and that tet count did not shrink below the uncut block.

## Adding tests

Prefer checks on real invariants (volume, manifold, neighbors, template tiling). Skip trivial getters. New GPU behavior that changes connectivity should extend the random-cut or mask suites rather than only adding viewer-side checks.
