# Tetrahedralizer

GPU tetrahedralization for unstructured, non-manifold polygonal meshes. The mesh does not need to be watertight or manifold: the algorithm voxelizes the surface, fills enclosed volume, and turns solid voxels into tetrahedra.

## Approach

1. **Voxelize** the input triangles onto a regular grid (`voxelSpacing`).
2. **Seal holes** (optional): morphological close (dilate then erode by `holeCloseRadius`) so small gaps in the shell do not leak during flood fill.
3. **Fill the interior**: stamp surface voxels into a dense bit grid, flood exterior air from the grid border, then keep every cell the flood did not reach (surface + enclosed solid).
4. **Tetrahedralize voxels**: each solid cell becomes five tets via an alternating five-tet cube decomposition so neighboring face diagonals agree.
5. **Smooth** (optional): shape-matching iterations that pull each tet toward a regular tet scaled by `volumeFactor`.

## Parameters

| Parameter | Role |
| --- | --- |
| `voxelSpacing` | Grid cell size in world units |
| `holeCloseRadius` | Morphological close radius in voxels before flood fill (`0` skips) |
| `numSmoothingIterations` | Shape-matching passes (`0` skips) |
| `volumeFactor` | Target regular-tet volume as a fraction of the current tet volume (`< 1` contracts) |
| `cutWithInputMesh` | When enabled, cut tet edges against the input surface and split tets with templates |

See `TetrahedralizerParams` in `include/tetrahedralizer/Tetrahedralizer.h`.

## Status

### Implemented

- Surface voxelization and unique solid-cell collection
- Optional hole closing and exterior flood-fill interior
- Five-tet voxel decomposition → node positions and tet indices
- Optional shape-matching smoothing
- Cut tets along the input mesh: BVH edge cuts, Steiner vertices, template-based tet split
- Face-neighbor table (`tet_neighbors`, 4 slots per tet) via sorted face pairing
- Interactive viewer: load OBJ, run tetrahedralization, inspect the tet mesh (clip planes, voxel size, smooth/cut options)

### Not implemented yet

- Project outer nodes onto the original surface
- Collapse short edges
- Split long edges

### Cut pipeline

Host-baked 64×diagBits child templates (from the tet-cut constructive builder). GPU passes stay separate:

1. **Cut vertices**: unique tet edges → BVH hits → append cut nodes; `edgeCutVertices[edge]`.
2. **Steiner vertices**: for each tet, resolve `mask` / `diagBits`; if the template uses local index 10, count → scan → append centroids; `steinerVertexId[tet]`.
3. **Split tets**: count → scan → write children from the template (`0..3` corners, `4..9` edge cuts, `10` Steiner).

## Layout

| Path | Role |
| --- | --- |
| `include/tetrahedralizer/Tetrahedralizer.h` | Public API and parameters |
| `src/Tetrahedralizer.cpp` | Host entry; dispatches to the GPU path |
| `src/tetrahedralizers/GpuTetrahedralizer.cu` | Voxelize → fill → tets → smooth → optional cut |
| `src/tetrahedralizers/TetCutTemplates.h` | Host-baked tet edge-cut subdivision tables |
| `src/utils/GpuBVH.*` | GPU BVH for edge–mesh queries |
| `src/main.cpp` | OpenGL / ImGui viewer |
| `src/TriMesh.*` | OBJ load |

## Build

Requires Visual Studio C++ tools, CMake, Ninja, and a CUDA toolkit (tetrahedralization is CUDA-only).

```bat
build.bat
```

This configures with Ninja Multi-Config and builds Debug and Release. Do not run bare `cmake --build` from a plain shell; the MSVC/CUDA environment will not be set up.

Run the viewer from `build\Debug` or `build\Release` (or `run.bat`).

## Tests

See [`TESTS.md`](TESTS.md). Quick run:

```bat
test.bat
```
