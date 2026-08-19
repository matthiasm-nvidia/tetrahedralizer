# Tetrahedralizer

GPU tetrahedralization for unstructured, non-manifold polygonal meshes. The mesh does not need to be watertight or manifold: the algorithm voxelizes the surface, fills enclosed volume, and turns solid voxels into tetrahedra.

## Approach

1. **Voxelize** the input triangles onto a regular grid (`voxelSpacing`).
2. **Seal holes** (optional): morphological close (dilate then erode by `holeCloseRadius`) so small gaps in the shell do not leak during flood fill.
3. **Fill the interior**: stamp surface voxels into a dense bit grid, flood exterior air from the grid border, then keep every cell the flood did not reach (surface + enclosed solid).
4. **Split voxels** (optional): only merge surface-voxel corners across faces crossed by input geometry; interior voxels remain connected.
5. **Tetrahedralize voxels**: each solid cell becomes five tets via an alternating five-tet cube decomposition so neighboring face diagonals agree.
6. **Subdivide** (optional): split tet edges longer than `maxEdgeLength` at their midpoints.
7. **Prepare boundary tets** (when projecting): separate multiple boundary faces into different tets.
8. **Optimize** (optional): each iteration runs one shape-matching smooth with tangential-only motion on surface nodes, then projects boundary nodes onto the input mesh (if enabled).

## Parameters

| Parameter | Role |
| --- | --- |
| `voxelSpacing` | Grid cell size in world units |
| `holeCloseRadius` | Morphological close radius in voxels before flood fill (`0` skips) |
| `splitVoxels` | Disconnect adjacent surface voxels when no input triangle crosses their shared face |
| `numOptimizationIterations` | Project (optional) + shape-matching smooth loops (`0` skips) |
| `volumeContraction` | Fraction of current tet volume to remove while shape-matching (`0` skips) |
| `edgeContraction` | Fraction of each tet-edge length to pull endpoints together (`0` skips) |
| `maxEdgeLength` | Subdivide tet edges longer than this value (`0` skips) |
| `projectToInputMesh` | Project boundary nodes onto the input surface (alone or inside the optimization loop) |

See `TetrahedralizerParams` in `include/tetrahedralizer/Tetrahedralizer.h`.

## Status

### Implemented

- Surface voxelization and unique solid-cell collection
- Optional hole closing and exterior flood-fill interior
- Optional face-aware voxel splitting before tet creation
- Five-tet voxel decomposition → node positions and tet indices
- Optional maximum-edge-length subdivision using midpoint vertices and template-based tet subdivision
- Optional projection of boundary nodes onto the input mesh (outward normals + BVH raycast)
- Optional optimization loop: shape-matching smooth then project
- Face-neighbor table (`tet_neighbors`, 4 slots per tet) via sorted face pairing
- Interactive viewer: load OBJ, run tetrahedralization, inspect the tet mesh (clip planes, voxel size, split/subdivision/project/optimize options)

### Not implemented yet

- Collapse short edges

### Voxel split pipeline

Enabled by `splitVoxels`. Runs after solid voxels are collected and before five-tet creation (`createSplitVoxelNodes` in `GpuTetrahedralizer.cu`). Goal: stop adjacent **surface** voxels from sharing nodes when no input geometry actually crosses their shared face (close sheets / touching shells that only meet in the grid).

Without splitting, all solid voxels share nodes by unique grid-corner coordinates, so any face-adjacent pair is topologically glued.

#### Surface vs interior

After flood fill, every non-air cell is solid. `geomCells` still holds the stamped (and optionally morphologically closed) **surface shell**. Classification:

- **Surface voxel**: bit set in `geomCells`
- **Interior voxel**: solid and not in `geomCells` (enclosed fill with no input triangles)

Interior faces must stay connected even though no triangles cross them.

#### Pass 1 — face connectivity

Solid voxel coords are sorted after collect so neighbor lookup (`findCoord`) is a binary search.

Array `connectedVoxelFaces[numVoxels * 6]`, face order `-X, +X, -Y, +Y, -Z, +Z` (`kVoxelFaceOffsets`).

For each solid voxel face:

1. Look up the 6-neighbor solid voxel by packed coords (`findCoord` on sorted `voxels`). No solid neighbor → disconnected.
2. If **either** side is interior → connected.
3. If **both** are surface → connected only if an input triangle intersects a thin AABB on that face:
   - Center = voxel center shifted by half a cell along the face normal onto the shared plane.
   - Extents = half-cell (+ `kVoxelOverlapMargin`) in the two tangential axes; thickness along the normal is only the margin (a flat “square” approximated as a thin box).
   - Query the input-mesh BVH; leaves tested with `boxTriangleIntersection`.

The BVH is built when `splitVoxels` or `projectToInputMesh` is on.

#### Pass 2 — corner IDs and merge

Local voxel corners (`kVoxelCorners`):

```
0 (0,0,0)  1 (1,0,0)  2 (1,1,0)  3 (0,1,0)
4 (0,0,1)  5 (1,0,1)  6 (1,1,1)  7 (0,1,1)
```

Each solid voxel starts with eight private corner IDs: `8 * voxelIndex + corner`.

Adjacent voxels do **not** use the same local corner index on a shared face, and face winding does not line up. Use the fixed pairing table `kVoxelFaceCornerPairs[face][0..3] = {localCorner, neighborLocalCorner}` (same world grid point). Example: +X pairs `1↔0, 2↔3, 5↔4, 6↔7`.

Parallel label propagation (same idea as smooth-vertex merge in mesh-tools-lib `SharpEdgeSplitter`):

1. `mergedCornerIds[i] = i`
2. For each connected face, for each of the four pairs: `atomicMin` both sides to `Min(id0, id1)`; set `anyChanged` if a label drops.
3. Repeat host-side until a pass makes no changes (labels reach component minima).

Then compact:

1. Mark used root labels → exclusive scan → dense node indices in `nodeOffsets`
2. Roots write world positions into `nodes`; every corner stores its dense id in `voxelCornerNodes[8 * voxel + corner]`

`createTets(..., splitVoxels=true)` reads those eight ids instead of looking up unique grid coords. Unconnected faces never share nodes, so the tet mesh stays disconnected there; connected faces share all four corners and the alternating five-tet pattern still matches diagonals.

#### Contrast with the default path

Default: `createCornerCoords` → sort/unique by grid position → one node per grid corner globally. Split: connectivity-limited union of per-voxel corners; same grid corner may become multiple nodes.

### Subdivision pipeline

Enabled when `maxEdgeLength > 0`. Host-baked 64×diagBits child templates provide conforming subdivisions for every combination of subdivided tet edges:

1. **Subdivision vertices**: unique tet edges longer than `maxEdgeLength` → append midpoint nodes.
2. **Steiner vertices**: for each tet, resolve `mask` / `diagBits`; if the template uses local index 10, count → scan → append centroids; `steinerVertexId[tet]`.
3. **Subdivide tets**: count → scan → write children from the template (`0..3` corners, `4..9` edge midpoints, `10` Steiner).

### Projection / optimization pipeline

Enabled by `projectToInputMesh` and/or `numOptimizationIterations > 0`. Runs after neighbors are built.

**Projection** (when `projectToInputMesh`):

Before projection, boundary topology is refined until no tet edge has both opposite faces on the boundary. Each such edge is split at its midpoint with the existing cut templates (a tet with two boundary faces splits one edge; three or four boundary faces mark every qualifying pair). Neighbors are rebuilt after each pass.

1. **Accumulate normals**: for each tet face with no neighbor, add the outward face normal to its three nodes.
2. **Normalize**: unit-length normals for non-zero accumulations; interior nodes stay zero.
3. **Raycast**: for each surface node, cast along `-normal` (fall back to `+normal`) against the input-mesh BVH; on hit, move to `hit + 0.1 * voxelSpacing * normal` (stay slightly outside). Hits farther than `2 * voxelSpacing` are ignored and the motion per pass is clamped to `0.1 * voxelSpacing`, so distant geometry cannot pull nodes into spikes. Misses leave the node unchanged.

Projection and smoothing never write node positions directly. Both fill a per-node offset buffer, and `applyNodeMovesSafely` then halves the step of every node belonging to a tet that the move would shrink below `kMinMoveVolumeFraction` of its current volume, repeating until no tet is affected (a step below `kMinMoveScale` becomes zero, which restores the original corners, so the loop always terminates). Without this backoff interior midpoints added by boundary refinement can cross a boundary face and produce spikes.

**Optimization** (`numOptimizationIterations` times): one shape-matching smooth, then edge contraction if enabled, then project if enabled. Surface nodes with a normal keep only the tangential part of the smooth correction (`corr -= n * dot(corr, n)`). With projection off, smooth runs unconstrained. With iterations `0` and projection on, projection runs once.

## Layout

| Path | Role |
| --- | --- |
| `include/tetrahedralizer/Tetrahedralizer.h` | Public API and parameters |
| `src/Tetrahedralizer.cpp` | Host entry; dispatches to the GPU path |
| `src/tetrahedralizers/GpuTetrahedralizer.cu` | Voxelize → fill → optional voxel split → tets → optional subdivision → optional optimize/project |
| `src/tetrahedralizers/TetCutTemplates.h` | Host-baked tet edge-subdivision tables |
| `src/utils/GpuBVH.*` | GPU BVH used by voxel face queries and projection raycasts |
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
