#include "TetDeviceData.h"

#include "utils/Geometry.h"
#include "utils/Math.h"

namespace tetrahedralizer
{
namespace
{

// Keep projected surface nodes slightly outside the input mesh.
constexpr float kProjectionOffsetFraction = 0.1f;
// Ignore surface hits farther away than this many voxels; they produce spikes.
constexpr float kProjectionMaxDistanceFraction = 2.0f;
// Largest distance a node may travel per projection pass.
constexpr float kProjectionMaxStepFraction = 0.1f;
// A node move may not leave a tet with less than this fraction of its volume.
constexpr float kMinMoveVolumeFraction = 0.2f;
// Below this fraction of the requested step a node simply stays where it is.
constexpr float kMinMoveScale = 1.0e-3f;

// Six times the volume of the regular tet returned by regularTetCorners.
constexpr float kRegularSixVolume = 16.0f;

// Regular tet centered at the origin with equal edge lengths and positive orientation.
__device__ void regularTetCorners(Vec3 q[4])
{
    q[0] = Vec3(1.0f, 1.0f, 1.0f);
    q[1] = Vec3(1.0f, -1.0f, -1.0f);
    q[2] = Vec3(-1.0f, -1.0f, 1.0f);
    q[3] = Vec3(-1.0f, 1.0f, -1.0f);
}

// Fit a rotated regular tet with remainingVolumeFraction * the current tet volume
// and accumulate the corner deltas.
__global__ void smoothAccumulate(TetDeviceData data, float remainingVolumeFraction)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    const int id0 = data.tetIndices[4 * tetIndex + 0];
    const int id1 = data.tetIndices[4 * tetIndex + 1];
    const int id2 = data.tetIndices[4 * tetIndex + 2];
    const int id3 = data.tetIndices[4 * tetIndex + 3];

    const Vec3 p0 = data.nodes[id0];
    const Vec3 p1 = data.nodes[id1];
    const Vec3 p2 = data.nodes[id2];
    const Vec3 p3 = data.nodes[id3];

    const Mat33 P(p1 - p0, p2 - p0, p3 - p0);
    const float sixVolume = P.getDeterminant();
    if (!(sixVolume > 1.0e-12f))
        return;

    Vec3 q[4];
    regularTetCorners(q);
    const Mat33 Q(q[1] - q[0], q[2] - q[0], q[3] - q[0]);
    const float targetScale = cbrtf(remainingVolumeFraction * sixVolume / kRegularSixVolume);

    Mat33 R, U, D;
    headerPolarDecomposition(P * Q.getInverse(), R, U, D);

    const Vec3 center = (p0 + p1 + p2 + p3) * 0.25f;
    const Vec3 targets[4] = {
        center + R * (q[0] * targetScale),
        center + R * (q[1] * targetScale),
        center + R * (q[2] * targetScale),
        center + R * (q[3] * targetScale),
    };
    const int ids[4] = {id0, id1, id2, id3};
    const Vec3 positions[4] = {p0, p1, p2, p3};
    for (int corner = 0; corner < 4; ++corner)
    {
        AtomicAdd(data.smoothOffsets.buffer + ids[corner], targets[corner] - positions[corner]);
        AtomicAdd(data.smoothCounts.buffer + ids[corner], 1);
    }
}

// Pull each tet edge's endpoints toward each other by contraction and accumulate the deltas.
// If normals is set, non-zero marks a surface node: same-type edges contract both ends,
// mixed edges move only the interior node.
__global__ void smoothAccumulateEdges(TetDeviceData data, float contraction, const Vec3* normals)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int ids[4];
    Vec3 positions[4];
    for (int corner = 0; corner < 4; ++corner)
    {
        ids[corner] = data.tetIndices[4 * tetIndex + corner];
        positions[corner] = data.nodes[ids[corner]];
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            const Vec3 delta = (positions[j] - positions[i]) * contraction;
            const bool surfaceI = normals && normals[ids[i]].magnitudeSquared() > 0.0f;
            const bool surfaceJ = normals && normals[ids[j]].magnitudeSquared() > 0.0f;
            if (!normals || surfaceI == surfaceJ)
            {
                AtomicAdd(data.smoothOffsets.buffer + ids[i], delta);
                AtomicAdd(data.smoothOffsets.buffer + ids[j], -delta);
                AtomicAdd(data.smoothCounts.buffer + ids[i], 1);
                AtomicAdd(data.smoothCounts.buffer + ids[j], 1);
            }
            else if (surfaceI)
            {
                AtomicAdd(data.smoothOffsets.buffer + ids[j], -delta);
                AtomicAdd(data.smoothCounts.buffer + ids[j], 1);
            }
            else
            {
                AtomicAdd(data.smoothOffsets.buffer + ids[i], delta);
                AtomicAdd(data.smoothCounts.buffer + ids[i], 1);
            }
        }
    }
}

__global__ void smoothApply(TetDeviceData data, const Vec3* normals)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    const int count = data.smoothCounts[nodeIndex];
    if (count == 0)
        return;

    Vec3 correction = data.smoothOffsets[nodeIndex] / static_cast<float>(count);
    if (normals)
    {
        const Vec3 normal = normals[nodeIndex];
        // Keep only tangential sliding on projected surface nodes.
        if (normal.magnitudeSquared() > 0.0f)
            correction -= normal * correction.dot(normal);
    }
    data.moveOffsets[nodeIndex] = correction;
}

__global__ void resetMoveScales(TetDeviceData data)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    data.moveScales[nodeIndex] = 1.0f;
}

__device__ float scaledSixVolume(const TetDeviceData& data, const int ids[4])
{
    Vec3 p[4];
    for (int corner = 0; corner < 4; ++corner)
    {
        const int id = ids[corner];
        p[corner] = data.nodes.buffer[id] + data.moveOffsets.buffer[id] * data.moveScales.buffer[id];
    }
    return Mat33(p[1] - p[0], p[2] - p[0], p[3] - p[0]).getDeterminant();
}

// Marks the nodes of every tet that the scaled moves would flatten or invert.
__global__ void flagCollapsingTets(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int ids[4];
    Vec3 p[4];
    for (int corner = 0; corner < 4; ++corner)
    {
        ids[corner] = data.tetIndices[4 * tetIndex + corner];
        p[corner] = data.nodes[ids[corner]];
    }

    const float sixVolume = Mat33(p[1] - p[0], p[2] - p[0], p[3] - p[0]).getDeterminant();
    // An already degenerate tet cannot be protected by backing off, so ignore it.
    if (!(sixVolume > 0.0f))
        return;
    if (scaledSixVolume(data, ids) >= kMinMoveVolumeFraction * sixVolume)
        return;

    for (int corner = 0; corner < 4; ++corner)
        data.moveBlocked[ids[corner]] = 1;
    data.anyChanged[0] = 1;
}

__global__ void shrinkBlockedMoves(TetDeviceData data)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    if (data.moveBlocked[nodeIndex] == 0)
        return;

    const float scale = data.moveScales[nodeIndex] * 0.5f;
    data.moveScales[nodeIndex] = scale > kMinMoveScale ? scale : 0.0f;
}

__global__ void applyNodeMoves(TetDeviceData data)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    data.nodes[nodeIndex] += data.moveOffsets[nodeIndex] * data.moveScales[nodeIndex];
}

// Halves the step of nodes whose tets would collapse until none does, then moves them.
// Zeroed scales restore the original corners, so the loop always reaches a safe state.
void applyNodeMovesSafely(TetDeviceData& data)
{
    constexpr int kMaxBackoffPasses = 20;

    data.moveScales.resize(static_cast<std::size_t>(data.numNodes), false);
    data.moveBlocked.resize(static_cast<std::size_t>(data.numNodes), false);
    data.anyChanged.resize(1, false);
    CUDA_LAUNCH(resetMoveScales, data.numNodes, data);

    for (int pass = 0; pass < kMaxBackoffPasses; ++pass)
    {
        data.moveBlocked.setZero();
        data.anyChanged.setZero();
        CUDA_LAUNCH(flagCollapsingTets, data.numTets, data);
        if (readDeviceInt(data.anyChanged, 0) == 0)
            break;
        CUDA_LAUNCH(shrinkBlockedMoves, data.numNodes, data);
    }

    CUDA_LAUNCH(applyNodeMoves, data.numNodes, data);
}

void smoothTets(TetDeviceData& data, int iterations, float volumeContraction, const Vec3* normals = nullptr)
{
    const float remainingVolumeFraction = 1.0f - volumeContraction;
    if (iterations <= 0 || !(volumeContraction > 0.0f) || !(remainingVolumeFraction > 0.0f) || data.numTets <= 0 ||
        data.numNodes <= 0)
        return;

    data.smoothOffsets.resize(static_cast<std::size_t>(data.numNodes));
    data.smoothCounts.resize(static_cast<std::size_t>(data.numNodes));
    data.moveOffsets.resize(static_cast<std::size_t>(data.numNodes), false);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        data.smoothOffsets.setZero();
        data.smoothCounts.setZero();
        data.moveOffsets.setZero();
        CUDA_LAUNCH(smoothAccumulate, data.numTets, data, remainingVolumeFraction);
        CUDA_LAUNCH(smoothApply, data.numNodes, data, normals);
        applyNodeMovesSafely(data);
    }
    cudaCheck(cudaDeviceSynchronize());
}

} // namespace

void smoothEdges(TetDeviceData& data, int iterations, float contraction, const Vec3* classifyNormals,
                 const Vec3* applyNormals)
{
    if (iterations <= 0 || !(contraction > 0.0f) || data.numTets <= 0 || data.numNodes <= 0)
        return;

    data.smoothOffsets.resize(static_cast<std::size_t>(data.numNodes));
    data.smoothCounts.resize(static_cast<std::size_t>(data.numNodes));
    data.moveOffsets.resize(static_cast<std::size_t>(data.numNodes), false);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        data.smoothOffsets.setZero();
        data.smoothCounts.setZero();
        data.moveOffsets.setZero();
        CUDA_LAUNCH(smoothAccumulateEdges, data.numTets, data, contraction, classifyNormals);
        CUDA_LAUNCH(smoothApply, data.numNodes, data, applyNormals);
        applyNodeMovesSafely(data);
    }
    cudaCheck(cudaDeviceSynchronize());
}

namespace
{

__global__ void accumulateSurfaceNormals(TetDeviceData data, Vec3* normals)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    const int* ids = data.tetIndices.buffer + 4 * tetIndex;
    for (int face = 0; face < 4; ++face)
    {
        if (data.tetNeighbors[4 * tetIndex + face] >= 0)
            continue;

        const int i0 = ids[kTetFaces[face][0]];
        const int i1 = ids[kTetFaces[face][1]];
        const int i2 = ids[kTetFaces[face][2]];
        const int iOpp = ids[6 - kTetFaces[face][0] - kTetFaces[face][1] - kTetFaces[face][2]];

        const Vec3 p0 = data.nodes[i0];
        Vec3 normal = (data.nodes[i1] - p0).cross(data.nodes[i2] - p0);
        // Face winding points toward the opposite vertex for a positive tet; flip for outward.
        if (normal.dot(data.nodes[iOpp] - p0) > 0.0f)
            normal = -normal;

        atomicAdd(&normals[i0].x, normal.x);
        atomicAdd(&normals[i0].y, normal.y);
        atomicAdd(&normals[i0].z, normal.z);
        atomicAdd(&normals[i1].x, normal.x);
        atomicAdd(&normals[i1].y, normal.y);
        atomicAdd(&normals[i1].z, normal.z);
        atomicAdd(&normals[i2].x, normal.x);
        atomicAdd(&normals[i2].y, normal.y);
        atomicAdd(&normals[i2].z, normal.z);
    }
}

__global__ void normalizeSurfaceNormals(Vec3* normals, int numNodes)
{
    CUDA_THREAD_GUARD(nodeIndex, numNodes)
    const Vec3 normal = normals[nodeIndex];
    const float lengthSq = normal.magnitudeSquared();
    if (lengthSq > 0.0f)
        normals[nodeIndex] = normal * (1.0f / sqrtf(lengthSq));
}

__device__ bool rayHitsAabb(const Ray& ray, float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
    float tEntry = -MaxFloat;
    float tExit = MaxFloat;
    const float origin[3] = {ray.orig.x, ray.orig.y, ray.orig.z};
    const float direction[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
    const float boundsMin[3] = {minX, minY, minZ};
    const float boundsMax[3] = {maxX, maxY, maxZ};

    for (int axis = 0; axis < 3; ++axis)
    {
        if (fabsf(direction[axis]) > 1.0e-20f)
        {
            const float t0 = (boundsMin[axis] - origin[axis]) / direction[axis];
            const float t1 = (boundsMax[axis] - origin[axis]) / direction[axis];
            tEntry = fmaxf(tEntry, fminf(t0, t1));
            tExit = fminf(tExit, fmaxf(t0, t1));
        }
        else if (origin[axis] < boundsMin[axis] || origin[axis] > boundsMax[axis])
        {
            return false;
        }
    }

    return tExit >= tEntry && tExit >= 0.0f;
}

// Unit ray direction, so maxT is a world distance.
__device__ bool raycastInputMesh(const TetDeviceData& data, const Ray& ray, float maxT, float& minT, int& minTri)
{
    int stack[64];
    stack[0] = data.triangleBvh.mRootNodes[0];
    int count = 1;
    minT = MaxFloat;
    minTri = -1;

    while (count > 0)
    {
        const int nodeIndex = stack[--count];
        const PackedNodeHalf lower = data.triangleBvh.mNodeLowers[nodeIndex];
        const PackedNodeHalf upper = data.triangleBvh.mNodeUppers[nodeIndex];
        if (!rayHitsAabb(ray, lower.x, lower.y, lower.z, upper.x, upper.y, upper.z))
            continue;

        if (lower.b)
        {
            const int triangleIndex = static_cast<int>(lower.i);
            const std::uint32_t i0 = data.meshIndices.buffer[3 * triangleIndex + 0];
            const std::uint32_t i1 = data.meshIndices.buffer[3 * triangleIndex + 1];
            const std::uint32_t i2 = data.meshIndices.buffer[3 * triangleIndex + 2];
            if (i0 >= data.meshVertices.size || i1 >= data.meshVertices.size || i2 >= data.meshVertices.size)
                continue;

            float t = 0.0f;
            float u = 0.0f;
            float v = 0.0f;
            if (header_rayTriangleIntersection(ray, data.meshVertices.buffer[i0], data.meshVertices.buffer[i1],
                                               data.meshVertices.buffer[i2], t, u, v) &&
                t >= 1.0e-6f && t <= maxT && t < minT)
            {
                minT = t;
                minTri = triangleIndex;
            }
        }
        else if (count <= 62)
        {
            stack[count++] = static_cast<int>(lower.i);
            stack[count++] = static_cast<int>(upper.i);
        }
    }

    return minTri >= 0;
}

__global__ void projectSurfaceNodes(TetDeviceData data, const Vec3* normals, float offset, float maxDistance,
                                    float maxStep)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)

    const Vec3 normal = normals[nodeIndex];
    if (normal.magnitudeSquared() == 0.0f)
        return;

    const Vec3 origin = data.nodes[nodeIndex];
    float minT = 0.0f;
    int minTri = -1;
    float distance = 0.0f;

    // Prefer the opposite of the outward normal (approach from outside); fall back to outward.
    if (raycastInputMesh(data, Ray(origin, -normal), maxDistance, minT, minTri))
        distance = offset - minT;
    else if (raycastInputMesh(data, Ray(origin, normal), maxDistance, minT, minTri))
        distance = minT + offset;
    else
        return;

    data.moveOffsets[nodeIndex] = normal * fminf(fmaxf(distance, -maxStep), maxStep);
}

__global__ void projectSurfaceNodesToClosestPoint(TetDeviceData data, const Vec3* normals)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)

    if (normals[nodeIndex].magnitudeSquared() == 0.0f)
        return;

    Vec3 query = data.nodes[nodeIndex];
    Vec3 bary;
    Vec3 closestPos;
    int closestTri = -1;
    bool inside = false;
    if (!header_queryClosestPoint(data.triangleBvh, query, 0.0f, data.meshVertices.buffer,
                                  reinterpret_cast<int*>(data.meshIndices.buffer), closestTri, bary, closestPos,
                                  inside) ||
        closestTri < 0)
        return;

    data.moveOffsets[nodeIndex] = closestPos - query;
}

void projectSurfaceNodesToInputMesh(TetDeviceData& data, const Vec3* normals, float voxelSpacing)
{
    if (data.numNodes <= 0 || data.numTets <= 0 || !normals)
        return;

    data.moveOffsets.resize(static_cast<std::size_t>(data.numNodes), false);
    data.moveOffsets.setZero();
    CUDA_LAUNCH(projectSurfaceNodes, data.numNodes, data, normals, kProjectionOffsetFraction * voxelSpacing,
                kProjectionMaxDistanceFraction * voxelSpacing, kProjectionMaxStepFraction * voxelSpacing);
    applyNodeMovesSafely(data);
}

void projectToClosestPointOnInputMesh(TetDeviceData& data, const Vec3* normals)
{
    if (data.numNodes <= 0 || data.numTets <= 0 || !normals)
        return;

    data.moveOffsets.resize(static_cast<std::size_t>(data.numNodes), false);
    data.moveOffsets.setZero();
    CUDA_LAUNCH(projectSurfaceNodesToClosestPoint, data.numNodes, data, normals);
    applyNodeMovesSafely(data);
}

void projectBoundaryNodes(TetDeviceData& data, const Vec3* normals, const TetrahedralizerParams& params)
{
    if (params.projectToClosestPoint)
        projectToClosestPointOnInputMesh(data, normals);
    else
        projectSurfaceNodesToInputMesh(data, normals, params.voxelSpacing);
}

} // namespace

void computeSurfaceNormals(TetDeviceData& data, DeviceBuffer<Vec3>& normals)
{
    normals.resize(static_cast<std::size_t>(data.numNodes), false);
    normals.setZero();
    CUDA_LAUNCH(accumulateSurfaceNormals, data.numTets, data, normals.buffer);
    CUDA_LAUNCH(normalizeSurfaceNormals, data.numNodes, normals.buffer, data.numNodes);
}

void runOptimization(TetDeviceData& data, const TetrahedralizerParams& params)
{
    const int totalIterations = params.numOptimizationIterations + params.numAdaptiveIterations;
    if (totalIterations <= 0)
        return;

    const bool project = params.projectToInputMesh;
    const bool canSplit = params.numAdaptiveIterations > 0 && data.meshVertexSizes.size > 0;
    const float maxSize = params.maxEdgeLength * params.voxelSpacing;
    DeviceBuffer<Vec3> normals;
    for (int iteration = 0; iteration < totalIterations; ++iteration)
    {
        if (canSplit && iteration >= params.numOptimizationIterations)
            runAdaptiveSplit(data, maxSize);

        const Vec3* smoothNormals = nullptr;
        if (project && params.volumeContraction > 0.0f && params.volumeContraction < 1.0f)
        {
            computeSurfaceNormals(data, normals);
            smoothNormals = normals.buffer;
        }
        smoothTets(data, 1, params.volumeContraction, smoothNormals);
        if (params.edgeContraction > 0.0f)
        {
            computeSurfaceNormals(data, normals);
            const Vec3* classifyNormals = params.useNormals ? nullptr : normals.buffer;
            const Vec3* applyNormals = params.useNormals ? normals.buffer : nullptr;
            smoothEdges(data, 1, params.edgeContraction, classifyNormals, applyNormals);
        }
        if (project)
        {
            computeSurfaceNormals(data, normals);
            projectBoundaryNodes(data, normals.buffer, params);
        }
    }
    normals.free();
}

} // namespace tetrahedralizer
