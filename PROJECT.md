# Tetrahedralizer

GPU tetrahedralization for unstructured, non-manifold polygonal meshes. The mesh does not need to be watertight or manifold: the algorithm voxelizes the surface, fills enclosed volume, and turns solid voxels into tetrahedra.

## Approach

1. **Voxelize** the input triangles onto a regular grid (`voxelSpacing`).
2. **Seal holes** (optional): morphological close (dilate then erode by `holeCloseRadius`) so small gaps in the shell do not leak during flood fill.
3. **Fill the interior**: stamp surface voxels into a dense bit grid, flood exterior air from the grid border, then keep every cell the flood did not reach (surface + enclosed solid).
4. **Split voxels**: only merge surface-voxel corners across faces crossed by input geometry; interior voxels remain connected.
5. **Tetrahedralize voxels**: each solid cell becomes five tets via an alternating five-tet cube decomposition so neighboring face diagonals agree.
6. **Adaptive remesh** (optional): when `adaptive` is on, split tet edges longer than the local size-field `h_max`.
7. **Prepare boundary tets**: separate multiple boundary faces into different tets.
8. **Optimize** (optional): each iteration runs tet smoothing, then edge smoothing, then projects boundary nodes onto the input mesh (if enabled).

## Parameters

| Parameter | Role |
| --- | --- |
| `voxelSpacing` | Grid cell size in world units |
| `holeCloseRadius` | Morphological close radius in voxels before flood fill (`0` skips) |
| `adaptive` | One-shot size-field remesh after tet creation |
| `minEdgeLength` | Floor for `h_max`, in voxel spacings (adaptive only). Default `0.5` |
| `maxEdgeLength` | Ceiling for `h_max`, in voxel spacings (adaptive only). Default `2.0` |
| `geometricError` | Max chordal error ε, in voxel spacings (adaptive only). Default `0.1` |
| `projectToInputMesh` | Project boundary nodes onto the input surface (alone or inside the optimization loop) |
| `projectToClosestPoint` | When projecting, snap to the closest input-mesh point instead of raycasting along the estimated normal |
| `numOptimizationIterations` | Tet smooth, then edge smooth, then project (if enabled). `0` skips the loop, including projection |
| `volumeContraction` | Fraction of current tet volume to remove while shape-matching (`0` skips tet smoothing) |
| `edgeContraction` | Fraction of each tet-edge length to pull endpoints together (`0` skips edge smoothing) |
| `useNormals` | Edge smoothing only. On: every edge contracts both ends, then surface nodes keep the tangential part. Off: mixed interior/surface edges move only the interior node |

See `TetrahedralizerParams` in `include/tetrahedralizer/Tetrahedralizer.h`. Adaptive length parameters are in voxel spacings; the pipeline multiplies by `voxelSpacing` before sampling the size field.

## Status

### Implemented

- Surface voxelization and unique solid-cell collection
- Optional hole closing and exterior flood-fill interior
- Face-aware voxel splitting before tet creation
- Five-tet voxel decomposition → node positions and tet indices
- One-shot adaptive remesh (`adaptive`): edge smooth, closest-point sample, split long edges until idle
- Optional projection of boundary nodes onto the input mesh (outward-normal raycast or closest point)
- Optional optimization loop: tet smoothing, edge smoothing, then project
- Face-neighbor table (`tet_neighbors`, 4 slots per tet) via sorted face pairing
- CPU curvature size field on the input mesh (`computeSurfaceSizeField`) and log-colormap visualization
- Interactive viewer: load OBJ, run tetrahedralization, inspect the tet mesh (clip planes, voxel size, subdivision/project/optimize options); left-click the input mesh to set the orbit point

### Not implemented yet

- Adaptive remesh inside optimize
- Adaptive collapse from the size field

### Voxel split pipeline

Always runs after solid voxels are collected and before five-tet creation (`createSplitVoxelNodes` in `VoxelSplit.cu`). Goal: stop adjacent **surface** voxels from sharing nodes when no input geometry actually crosses their shared face (close sheets / touching shells that only meet in the grid).

Without this merge limit, all solid voxels would share nodes by unique grid-corner coordinates, so any face-adjacent pair would be topologically glued. The same grid corner may become multiple nodes.

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
   - Query the input-mesh BVH; leaves tested with `header_boxTriangleIntersection`.

The BVH is built for every run (voxel-face queries, and later projection if enabled).

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

`createTets` reads those eight ids. Unconnected faces never share nodes, so the tet mesh stays disconnected there; connected faces share all four corners and the alternating five-tet pattern still matches diagonals.

### Subdivision pipeline

Enabled by `adaptive` remesh and by `Tetrahedralizer::subdivide`. Host-baked 64×diagBits child templates provide conforming subdivisions for every combination of subdivided tet edges:

1. **Subdivision vertices**: unique tet edges longer than `maxEdgeLength` → append midpoint nodes.
2. **Steiner vertices**: for each tet, resolve `mask` / `diagBits`; if the template uses local index 10, count → scan → append centroids; `steinerVertexId[tet]`.
3. **Subdivide tets**: count → scan → write children from the template (`0..3` corners, `4..9` edge midpoints, `10` Steiner).

### Projection / optimization pipeline

Enabled when `numOptimizationIterations > 0`. Runs after neighbors are built.

**Projection** (when `projectToInputMesh`):

Before projection, boundary topology is refined until no tet edge has both opposite faces on the boundary. Each such edge is split at its midpoint with the existing cut templates (a tet with two boundary faces splits one edge; three or four boundary faces mark every qualifying pair). Neighbors are rebuilt after each pass.

1. **Accumulate normals**: for each tet face with no neighbor, add the outward face normal to its three nodes.
2. **Normalize**: unit-length normals for non-zero accumulations; interior nodes stay zero.
3. **Snap**: with `projectToClosestPoint` off, cast along `-normal` (fall back to `+normal`) against the input-mesh BVH; on hit, move to `hit + 0.1 * voxelSpacing * normal` (stay slightly outside). Hits farther than `2 * voxelSpacing` are ignored and the motion per pass is clamped to `0.1 * voxelSpacing`. Misses leave the node unchanged. With `projectToClosestPoint` on, snap to the closest input-mesh point instead (no offset; may move inward or outward).

Projection and smoothing never write node positions directly. Both fill a per-node offset buffer, and `applyNodeMovesSafely` then halves the step of every node belonging to a tet that the move would shrink below `kMinMoveVolumeFraction` of its current volume, repeating until no tet is affected (a step below `kMinMoveScale` becomes zero, which restores the original corners, so the loop always terminates). Without this backoff interior midpoints added by boundary refinement can cross a boundary face and produce spikes.

**Optimization** (`numOptimizationIterations` times): tet smoothing if `volumeContraction > 0`, then edge smoothing if `edgeContraction > 0`, then project if enabled. Tet smoothing shape-matches to a regular tet. With projection on, surface nodes keep only the tangential part of the tet-smooth correction (`corr -= n * dot(corr, n)`). Edge smoothing with `useNormals` on contracts every edge then strips the normal component on surface nodes; with it off, mixed interior/surface edges move only the interior node. Iterations `0` skips the loop; projection does not run.

### Adaptive mesh

The size field and input-mesh visualization are implemented. When `adaptive` is on, a one-shot remesh runs after tet creation: a few edge-smooth steps, closest-point size sample on surface nodes, then split edges longer than local `h_max` until idle. `minEdgeLength` / `maxEdgeLength` / `geometricError` are voxel-relative clamps and ε; they are ignored when adaptive is off. Optimize does not change topology.

Goal: a spatially varying surface edge length so split/collapse capture input curvature, instead of those globals.

Uniform one-shot split then collapse before optimize is enough on the voxel grid. It is not enough once nodes sit on the real surface: projection lengthens edges on convex regions and shortens them on concave ones, and both split and surface–surface collapse place nodes at chord midpoints that are off the surface until project runs. Adaptive sizing should live in the optimize loop.

#### Size field on the input mesh

`computeSurfaceSizeField` (`src/SizeField.cpp`) stores one scalar per **input-mesh vertex** (parallel to `positions`): the local maximum surface edge length `h_max`. CPU only. Tet nodes do not own the field; they will sample it later. A tet edge uses the **minimum** of its endpoint samples (or the minimum of a triangle's three vertices) so the stricter vertex wins.

Do not use mean curvature. It can vanish at a saddle while the surface still bends. Use **maximum turning** (max |principal curvature| κ), from vertex-normal variation, which also works on non-manifold input:

1. Area-weighted vertex normals from incident input faces.
2. For each input edge `(i, j)` of length `L`, with unit normals `n_i`, `n_j`: turning angle `α = angle(n_i, n_j)`, then `κ = 2 sin(α/2) / L` (osculating circle). `R = 1/κ`.
3. At each vertex, take the **maximum** κ over incident edges.

The user-facing parameter is a geometric error `ε` (max distance from a surface chord to the input), not a global edge length. After projection, a surface tet edge is a chord of the input. For radius `R`:

- `h ≤ 2 √(2 R ε − ε²)` if `ε < R`
- `h ≈ √(8 R ε)` when `ε ≪ R`

So `h_max = clamp(√(8 ε / κ), h_min, h_max_global)` with κ floored so flat regions do not explode. Practical clamps: floor around `voxelSpacing` (the voxel tet mesh cannot resolve below that until you subdivide); ceiling at the current coarse size so interiors and flat walls stay cheap. Interior tet edges keep a large cap; curvature does not apply there.

Isotropic sizing from |κ|_max is the first version (a cylinder is over-refined along the axis). Anisotropic sizing, local feature size / medial axis, and a posteriori chordal-error refinement are later.

#### Smoothing

Tessellation noise spikes κ. Smooth **`h_max`**, not κ: averaging curvature and then taking `√(1/κ)` still maps a leftover spike to a tiny edge. Same accumulate/apply pattern as tet and edge smoothing, a few iterations (about 2–5); more washes out real features. The CPU path does this sequentially:

1. **Accumulate** (per input triangle): mean of the three vertex values, add that mean onto each vertex and add 1 onto each count.
2. **Apply** (per vertex): `value = sum / count`.

The triangle mean includes the vertex itself, so each step keeps about a third of the old value. Area-weighted `atomicAdd(mean * area)` / `count += area` is optional. To preserve creases later, lock vertices with a large dihedral rather than mixing the unsmoothed field back in.

#### Visualization

The viewer colors the **input mesh**, not the tets (`Show size field`). Color `h_max` (log-scaled colormap, range slider): long/blue = flat, short/red = high curvature. Salt-and-pepper means the smoother needs more iterations or the input has folded triangles.

#### Remesh inside optimize

Once the field looks right, drive split/collapse with the local threshold instead of a scalar. Lookup: closest input triangle (BVH / closest-point, min of its vertices) or splat `min(h)` into surface voxels during voxelization.

Put topology at the **start** of each optimize iteration so the existing end-of-iteration project snaps new nodes back. Last iteration must end with project, not with split/collapse. Hysteresis so split and collapse do not fight (`h_min ≈ 0.4–0.5` of local `h_max`):

1. Split surface tet edges longer than local `h_max`.
2. Collapse edges shorter than local `h_min` (surface–surface → midpoint; mixed → keep the surface node).
3. Rebuild neighbors and **separate boundary faces** (split and collapse can create tets with two opposite boundary faces; projection still forbids that).
4. Tet smooth, then edge smooth.
5. Project.

Do not project after every collapse kernel; once per iteration after topology has settled is enough. Interior collapses do not need it. Topology every iteration is not required: every *k* iterations, or until no splits/collapses fire, is fine, but geometry still needs a project after the last topology change.

## Layout

| Path | Role |
| --- | --- |
| `include/tetrahedralizer/Tetrahedralizer.h` | Public API and parameters |
| `include/tetrahedralizer/SizeField.h` | CPU size field (`h_max` from max turning) |
| `include/tetrahedralizer/TriMesh.h` | OBJ load |
| `src/SizeField.cpp` | Host size-field compute and smoothing |
| `src/tetrahedralizers/AdaptiveRemesh.cu` | One-shot adaptive remesh (smooth, sample, collapse/split) |
| `src/Tetrahedralizer.cpp` | Host entry; dispatches to the GPU path |
| `src/tetrahedralizers/GpuTetrahedralizer.cu` | Pipeline orchestrator (`create` / `subdivide`) |
| `src/tetrahedralizers/Voxelize.cu` | Surface voxelization, hole close, flood fill |
| `src/tetrahedralizers/VoxelSplit.cu` | Face-aware corner merge and five-tet cubes |
| `src/tetrahedralizers/TetCut.cu` | Neighbors, long-edge subdivision, boundary face splits |
| `src/tetrahedralizers/Optimize.cu` | Shape-matching smooth, edge smooth, project |
| `src/tetrahedralizers/TetCutTemplates.h` | Host-baked tet edge-subdivision tables |
| `src/utils/GpuBVH.*` | GPU BVH used by voxel face queries and projection raycasts |
| `src/utils/CpuBVH.*` | Host BVH for viewer picking (click-to-orbit) |
| `src/viewer/` | OpenGL / ImGui viewer |

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
