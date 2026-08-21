#include "TetDeviceData.h"

#include "utils/Math.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace tetrahedralizer
{
namespace
{

// Orientation restrict is implemented but not launched. Enable if inverted tets show up.
constexpr bool kRestrictCollapses = false;

struct TetKey
{
    int id0 = 0;
    int id1 = 0;
    int id2 = 0;
    int id3 = 0;
    int tetIndex = 0;

    __host__ __device__ bool sameVerts(const TetKey& other) const
    {
        return id0 == other.id0 && id1 == other.id1 && id2 == other.id2 && id3 == other.id3;
    }
};

struct TetKeyComparator
{
    __host__ __device__ bool operator()(const TetKey& a, const TetKey& b) const
    {
        if (a.id0 != b.id0)
            return a.id0 < b.id0;
        if (a.id1 != b.id1)
            return a.id1 < b.id1;
        if (a.id2 != b.id2)
            return a.id2 < b.id2;
        if (a.id3 != b.id3)
            return a.id3 < b.id3;
        return a.tetIndex < b.tetIndex;
    }
};

__device__ int hashPriority(int x)
{
    unsigned int u = static_cast<unsigned int>(x);
    u *= 0x45d9f3bu;
    u ^= u >> 16;
    return static_cast<int>(u & 0x7FFFFFFFu) | 1;
}

__device__ void sort4(int& a, int& b, int& c, int& d)
{
    auto swapIf = [](int& x, int& y) {
        if (x > y)
        {
            const int tmp = x;
            x = y;
            y = tmp;
        }
    };
    swapIf(a, b);
    swapIf(c, d);
    swapIf(a, c);
    swapIf(b, d);
    swapIf(b, c);
}

__device__ float tetSixVolume(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3)
{
    return Mat33(p1 - p0, p2 - p0, p3 - p0).getDeterminant();
}

__global__ void computeValences(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    atomicAdd(data.smoothCounts.buffer + data.tetIndices[4 * tetIndex + 0], 1);
    atomicAdd(data.smoothCounts.buffer + data.tetIndices[4 * tetIndex + 1], 1);
    atomicAdd(data.smoothCounts.buffer + data.tetIndices[4 * tetIndex + 2], 1);
    atomicAdd(data.smoothCounts.buffer + data.tetIndices[4 * tetIndex + 3], 1);
}

__global__ void createCollapseMap(TetDeviceData data, float minEdgeLength, bool lockEdges, const Vec3* normals)
{
    CUDA_THREAD_GUARD(globalEdgeNr, data.numTets * 6)

    const int tetIndex = globalEdgeNr / 6;
    const int edgeIndex = globalEdgeNr % 6;
    const int id0 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][0]];
    const int id1 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][1]];

    const Vec3 p0 = data.nodes[id0];
    const Vec3 p1 = data.nodes[id1];
    if ((p1 - p0).lengthSquared() > minEdgeLength * minEdgeLength)
        return;

    const int priority = hashPriority(globalEdgeNr + 1);
    if (lockEdges)
    {
        atomicMax(data.moveBlocked.buffer + id0, priority);
        atomicMax(data.moveBlocked.buffer + id1, priority);
        return;
    }

    if (data.moveBlocked[id0] != priority || data.moveBlocked[id1] != priority)
        return;

    const bool surface0 = normals[id0].magnitudeSquared() > 0.0f;
    const bool surface1 = normals[id1].magnitudeSquared() > 0.0f;
    if (surface0 != surface1)
    {
        if (surface0)
            data.collapseMap[id1] = id0 + 1;
        else
            data.collapseMap[id0] = id1 + 1;
        data.anyChanged[0] = 1;
        return;
    }

    const int val0 = data.smoothCounts[id0];
    const int val1 = data.smoothCounts[id1];
    const Vec3 newPosition = p0 + (p1 - p0) * 0.5f;

    if (val0 > val1)
    {
        data.collapseMap[id1] = id0 + 1;
        data.collapsePositions[id0] = newPosition;
    }
    else if (val1 > val0)
    {
        data.collapseMap[id0] = id1 + 1;
        data.collapsePositions[id1] = newPosition;
    }
    else
    {
        data.collapseMap[id1] = id0 + 1;
        data.collapsePositions[id0] = newPosition;
    }
    data.anyChanged[0] = 1;
}

__global__ void restrictCollapseMap(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    const int id0 = data.tetIndices[4 * tetIndex + 0];
    const int id1 = data.tetIndices[4 * tetIndex + 1];
    const int id2 = data.tetIndices[4 * tetIndex + 2];
    const int id3 = data.tetIndices[4 * tetIndex + 3];

    const int newId0 = data.collapseMap[id0] > 0 ? data.collapseMap[id0] - 1 : id0;
    const int newId1 = data.collapseMap[id1] > 0 ? data.collapseMap[id1] - 1 : id1;
    const int newId2 = data.collapseMap[id2] > 0 ? data.collapseMap[id2] - 1 : id2;
    const int newId3 = data.collapseMap[id3] > 0 ? data.collapseMap[id3] - 1 : id3;

    if (newId0 == id0 && newId1 == id1 && newId2 == id2 && newId3 == id3)
        return;
    if (newId0 == newId1 || newId0 == newId2 || newId0 == newId3 || newId1 == newId2 || newId1 == newId3 ||
        newId2 == newId3)
        return;

    const float oldSix = tetSixVolume(data.nodes[id0], data.nodes[id1], data.nodes[id2], data.nodes[id3]);
    const float newSix = tetSixVolume(data.collapsePositions[newId0], data.collapsePositions[newId1],
                                      data.collapsePositions[newId2], data.collapsePositions[newId3]);
    if (oldSix * newSix > 0.0f)
        return;

    data.collapseMap[id0] = 0;
    data.collapseMap[id1] = 0;
    data.collapseMap[id2] = 0;
    data.collapseMap[id3] = 0;
    data.collapsePositions[id0] = data.nodes[id0];
    data.collapsePositions[id1] = data.nodes[id1];
    data.collapsePositions[id2] = data.nodes[id2];
    data.collapsePositions[id3] = data.nodes[id3];
    atomicAdd(data.anyChanged.buffer, 1);
}

struct CollapseFace
{
    int id0 = 0;
    int id1 = 0;
    int id2 = 0;
    int tetIndex = 0;

    __host__ __device__ bool sameVerts(const CollapseFace& other) const
    {
        return id0 == other.id0 && id1 == other.id1 && id2 == other.id2;
    }
};

struct CollapseFaceComparator
{
    __host__ __device__ bool operator()(const CollapseFace& a, const CollapseFace& b) const
    {
        if (a.id0 != b.id0)
            return a.id0 < b.id0;
        if (a.id1 != b.id1)
            return a.id1 < b.id1;
        if (a.id2 != b.id2)
            return a.id2 < b.id2;
        return a.tetIndex < b.tetIndex;
    }
};

__device__ void sort3(int& a, int& b, int& c)
{
    if (a > b)
    {
        const int tmp = a;
        a = b;
        b = tmp;
    }
    if (b > c)
    {
        const int tmp = b;
        b = c;
        c = tmp;
    }
    if (a > b)
    {
        const int tmp = a;
        a = b;
        b = tmp;
    }
}

__device__ int vertexOutsideFace(const int* tetIds, int id0, int id1, int id2)
{
    for (int corner = 0; corner < 4; ++corner)
    {
        const int id = tetIds[corner];
        if (id != id0 && id != id1 && id != id2)
            return id;
    }
    return -1;
}

__global__ void applyCollapseMap(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int ids[4];
    for (int corner = 0; corner < 4; ++corner)
    {
        const int id = data.tetIndices[4 * tetIndex + corner];
        ids[corner] = data.collapseMap[id] > 0 ? data.collapseMap[id] - 1 : id;
        data.tetIndices[4 * tetIndex + corner] = ids[corner];
    }

    data.firstNewTet[tetIndex] =
        (ids[0] == ids[1] || ids[0] == ids[2] || ids[0] == ids[3] || ids[1] == ids[2] || ids[1] == ids[3] ||
         ids[2] == ids[3])
            ? 0
            : 1;
}

__global__ void fillCollapseFaces(TetDeviceData data, CollapseFace* faces)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    if (data.firstNewTet[tetIndex] == 0)
    {
        for (int face = 0; face < 4; ++face)
            faces[4 * tetIndex + face] = CollapseFace{-1 - tetIndex, -1, -1, tetIndex};
        return;
    }

    for (int face = 0; face < 4; ++face)
    {
        int id0 = data.tetIndices[4 * tetIndex + kTetFaces[face][0]];
        int id1 = data.tetIndices[4 * tetIndex + kTetFaces[face][1]];
        int id2 = data.tetIndices[4 * tetIndex + kTetFaces[face][2]];
        sort3(id0, id1, id2);
        faces[4 * tetIndex + face] = CollapseFace{id0, id1, id2, tetIndex};
    }
}

__device__ int faceApex(const int* tetIndices, int tetIndex, int id0, int id1, int id2)
{
    return vertexOutsideFace(tetIndices + 4 * tetIndex, id0, id1, id2);
}

__global__ void unmarkInvalidCollapseFaces(int numFaces, const CollapseFace* faces, const int* tetIndices,
                                           int* tetMarks, bool onlySameApex)
{
    CUDA_THREAD_GUARD(faceIndex, numFaces)

    const CollapseFace face0 = faces[faceIndex];
    if (faceIndex > 0 && face0.sameVerts(faces[faceIndex - 1]))
        return;
    if (face0.id0 < 0)
        return;

    int end = faceIndex + 1;
    while (end < numFaces && face0.sameVerts(faces[end]))
        ++end;
    const int count = end - faceIndex;

    if (onlySameApex)
    {
        for (int i = faceIndex; i < end; ++i)
        {
            const int apexI = faceApex(tetIndices, faces[i].tetIndex, face0.id0, face0.id1, face0.id2);
            if (apexI < 0)
                continue;
            for (int j = i + 1; j < end; ++j)
            {
                if (apexI == faceApex(tetIndices, faces[j].tetIndex, face0.id0, face0.id1, face0.id2))
                {
                    tetMarks[faces[i].tetIndex] = 0;
                    tetMarks[faces[j].tetIndex] = 0;
                }
            }
        }
        return;
    }

    if (count > 2)
    {
        for (int i = faceIndex; i < end; ++i)
            tetMarks[faces[i].tetIndex] = 0;
    }
}

__global__ void fillTetKeys(TetDeviceData data, TetKey* keys)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    if (data.firstNewTet[tetIndex] == 0)
    {
        keys[tetIndex] = TetKey{-1, -1, -1, -1, tetIndex};
        return;
    }

    int id0 = data.tetIndices[4 * tetIndex + 0];
    int id1 = data.tetIndices[4 * tetIndex + 1];
    int id2 = data.tetIndices[4 * tetIndex + 2];
    int id3 = data.tetIndices[4 * tetIndex + 3];
    sort4(id0, id1, id2, id3);
    keys[tetIndex] = TetKey{id0, id1, id2, id3, tetIndex};
}

__global__ void unmarkDuplicateTets(int numTets, const TetKey* keys, int* tetMarks)
{
    CUDA_THREAD_GUARD(index, numTets)

    const TetKey key0 = keys[index];
    if (index > 0 && key0.sameVerts(keys[index - 1]))
        return;
    if (key0.id0 < 0)
        return;

    int next = index + 1;
    int duplicates = 0;
    while (next < numTets && key0.sameVerts(keys[next]))
    {
        tetMarks[keys[next].tetIndex] = 0;
        ++duplicates;
        ++next;
    }
    if (duplicates > 0)
        tetMarks[key0.tetIndex] = 0;
}

__global__ void markUsedNodes(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    if (data.firstNewTet[tetIndex] == 0)
        return;
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 0], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 1], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 2], 1);
    atomicOr(data.firstSteiner.buffer + data.tetIndices[4 * tetIndex + 3], 1);
}

__global__ void compressNodes(TetDeviceData data, Vec3* compressedNodes)
{
    CUDA_THREAD_GUARD(nodeIndex, data.numNodes)
    const int mappedId = data.firstSteiner[nodeIndex];
    if (mappedId == data.firstSteiner[nodeIndex + 1])
        return;
    compressedNodes[mappedId] = data.nodes[nodeIndex];
}

__global__ void compressTetIds(TetDeviceData data, int* compressedTetIds)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)
    const int mappedId = data.firstNewTet[tetIndex];
    if (mappedId == data.firstNewTet[tetIndex + 1])
        return;
    compressedTetIds[4 * mappedId + 0] = data.firstSteiner[data.tetIndices[4 * tetIndex + 0]];
    compressedTetIds[4 * mappedId + 1] = data.firstSteiner[data.tetIndices[4 * tetIndex + 1]];
    compressedTetIds[4 * mappedId + 2] = data.firstSteiner[data.tetIndices[4 * tetIndex + 2]];
    compressedTetIds[4 * mappedId + 3] = data.firstSteiner[data.tetIndices[4 * tetIndex + 3]];
}

void unmarkTetsByRemappedFaces(TetDeviceData& data, bool onlySameApex)
{
    if (data.numTets > std::numeric_limits<int>::max() / 4)
        throw std::runtime_error("Tet face count exceeds the supported range");

    const int numFaces = data.numTets * 4;
    DeviceBuffer<CollapseFace> faces;
    faces.resize(static_cast<std::size_t>(numFaces), false);
    CUDA_LAUNCH(fillCollapseFaces, data.numTets, data, faces.buffer);
    thrust::device_ptr<CollapseFace> facePtr(faces.buffer);
    thrust::sort(facePtr, facePtr + numFaces, CollapseFaceComparator());
    CUDA_LAUNCH(unmarkInvalidCollapseFaces, numFaces, numFaces, faces.buffer, data.tetIndices.buffer,
                data.firstNewTet.buffer, onlySameApex);
    faces.free();
}

void restrictCollapses(TetDeviceData& data)
{
    constexpr int kMaxIterations = 100;
    data.anyChanged.resize(1, false);
    for (int iter = 0; iter < kMaxIterations; ++iter)
    {
        data.anyChanged.setZero();
        CUDA_LAUNCH(restrictCollapseMap, data.numTets, data);
        if (readDeviceInt(data.anyChanged, 0) == 0)
            break;
    }
}

void compactTets(TetDeviceData& data)
{
    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstSteiner.resize(static_cast<std::size_t>(data.numNodes + 1), false);
    data.firstSteiner.setZero();
    CUDA_LAUNCH(markUsedNodes, data.numTets, data);

    thrust::device_ptr<int> tetMap(data.firstNewTet.buffer);
    thrust::device_ptr<int> nodeMap(data.firstSteiner.buffer);
    thrust::exclusive_scan(tetMap, tetMap + data.numTets + 1, tetMap);
    thrust::exclusive_scan(nodeMap, nodeMap + data.numNodes + 1, nodeMap);

    const int numNewTets = readDeviceInt(data.firstNewTet, data.numTets);
    const int numNewNodes = readDeviceInt(data.firstSteiner, data.numNodes);

    DeviceBuffer<Vec3> compressedNodes;
    DeviceBuffer<int> compressedTetIds;
    compressedNodes.resize(static_cast<std::size_t>(numNewNodes), false);
    compressedTetIds.resize(static_cast<std::size_t>(numNewTets) * 4, false);
    CUDA_LAUNCH(compressNodes, data.numNodes, data, compressedNodes.buffer);
    CUDA_LAUNCH(compressTetIds, data.numTets, data, compressedTetIds.buffer);

    data.nodes.swap(compressedNodes);
    data.tetIndices.swap(compressedTetIds);
    data.numNodes = numNewNodes;
    data.numTets = numNewTets;
    compressedNodes.free();
    compressedTetIds.free();
}

} // namespace

bool collapseShortEdges(TetDeviceData& data, float minEdgeLength)
{
    if (!(minEdgeLength > 0.0f) || data.numTets <= 0 || data.numNodes <= 0)
        return false;
    if (data.numTets > std::numeric_limits<int>::max() / 6)
        throw std::runtime_error("Tet edge count exceeds the supported range");
    if (!computeNeighbors(data))
        throw std::runtime_error("Tet mesh has non-manifold faces before edge collapse");

    DeviceBuffer<Vec3> normals;
    computeSurfaceNormals(data, normals);

    data.smoothCounts.resize(static_cast<std::size_t>(data.numNodes), false);
    data.smoothCounts.setZero();
    CUDA_LAUNCH(computeValences, data.numTets, data);

    data.collapsePositions.resize(static_cast<std::size_t>(data.numNodes), false);
    cudaCheck(cudaMemcpy(data.collapsePositions.buffer, data.nodes.buffer,
                         static_cast<std::size_t>(data.numNodes) * sizeof(Vec3), cudaMemcpyDeviceToDevice));

    data.collapseMap.resize(static_cast<std::size_t>(data.numNodes), false);
    data.collapseMap.setZero();
    data.moveBlocked.resize(static_cast<std::size_t>(data.numNodes), false);
    data.moveBlocked.setZero();
    data.anyChanged.resize(1, false);
    data.anyChanged.setZero();

    const int numEdges = data.numTets * 6;
    CUDA_LAUNCH(createCollapseMap, numEdges, data, minEdgeLength, true, normals.buffer);
    CUDA_LAUNCH(createCollapseMap, numEdges, data, minEdgeLength, false, normals.buffer);
    normals.free();
    if (readDeviceInt(data.anyChanged, 0) == 0)
        return false;

    if (kRestrictCollapses)
        restrictCollapses(data);

    data.nodes.swap(data.collapsePositions);

    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstNewTet.setZero();
    CUDA_LAUNCH(applyCollapseMap, data.numTets, data);
    // Same-apex pairs first so a glued "blue" pair is removed before its side
    // faces are treated as overused (those side faces would otherwise delete the
    // outer tets too). Then drop any face still used by more than two tets.
    unmarkTetsByRemappedFaces(data, true);
    unmarkTetsByRemappedFaces(data, false);

    DeviceBuffer<TetKey> keys;
    keys.resize(static_cast<std::size_t>(data.numTets), false);
    CUDA_LAUNCH(fillTetKeys, data.numTets, data, keys.buffer);
    thrust::device_ptr<TetKey> keyPtr(keys.buffer);
    thrust::sort(keyPtr, keyPtr + data.numTets, TetKeyComparator());
    CUDA_LAUNCH(unmarkDuplicateTets, data.numTets, data.numTets, keys.buffer, data.firstNewTet.buffer);
    keys.free();

    compactTets(data);
    return true;
}

} // namespace tetrahedralizer
