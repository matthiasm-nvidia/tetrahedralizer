#include "TetDeviceData.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace tetrahedralizer
{
namespace
{

struct Face
{
    int id0 = 0;
    int id1 = 0;
    int id2 = 0;
    int faceNr = 0; // 4 * tetIndex + localFace

    __host__ __device__ bool operator==(const Face& other) const
    {
        return id0 == other.id0 && id1 == other.id1 && id2 == other.id2;
    }
};

struct FaceComparator
{
    __host__ __device__ bool operator()(const Face& a, const Face& b) const
    {
        if (a.id0 != b.id0)
            return a.id0 < b.id0;
        if (a.id1 != b.id1)
            return a.id1 < b.id1;
        return a.id2 < b.id2;
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

__host__ __device__ std::uint64_t packEdge(int id0, int id1)
{
    const std::uint32_t lower = static_cast<std::uint32_t>(id0 < id1 ? id0 : id1);
    const std::uint32_t upper = static_cast<std::uint32_t>(id0 < id1 ? id1 : id0);
    return (static_cast<std::uint64_t>(lower) << 32) | upper;
}

__device__ int findPackedEdge(const DeviceBuffer<std::uint64_t>& edges, std::uint64_t target)
{
    int first = 0;
    int last = static_cast<int>(edges.size);
    while (first < last)
    {
        const int middle = first + (last - first) / 2;
        if (edges.buffer[middle] < target)
            first = middle + 1;
        else
            last = middle;
    }
    if (first >= static_cast<int>(edges.size) || edges.buffer[first] != target)
        return -1;
    return first;
}

__global__ void createTetEdges(TetDeviceData data)
{
    CUDA_THREAD_GUARD(index, data.numTets * 6)
    const int tetIndex = index / 6;
    const int edgeIndex = index % 6;
    const int id0 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][0]];
    const int id1 = data.tetIndices[4 * tetIndex + kTetEdges[edgeIndex][1]];
    data.edges[index] = packEdge(id0, id1);
}

__global__ void fillFaces(TetDeviceData data, Face* faces)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    for (int face = 0; face < 4; ++face)
    {
        int id0 = data.tetIndices[4 * tetIndex + kTetFaces[face][0]];
        int id1 = data.tetIndices[4 * tetIndex + kTetFaces[face][1]];
        int id2 = data.tetIndices[4 * tetIndex + kTetFaces[face][2]];
        sort3(id0, id1, id2);

        const int faceNr = 4 * tetIndex + face;
        faces[faceNr] = Face{id0, id1, id2, faceNr};
        data.tetNeighbors[faceNr] = -1;
    }
}

// Pairs faces that share the same three vertices. More than two tets on one face
// increments nonManifold (same pattern as triangle edge neighbors in mesh-tools-lib).
__global__ void fillNeighbors(int numFaces, const Face* faces, int* neighbors, int* nonManifold)
{
    CUDA_THREAD_GUARD(faceIndex, numFaces)

    const Face face0 = faces[faceIndex];
    if (faceIndex > 0 && face0 == faces[faceIndex - 1])
        return;

    int num = 0;
    int next = faceIndex + 1;
    while (next < numFaces && face0 == faces[next])
    {
        const Face face1 = faces[next];
        neighbors[face0.faceNr] = face1.faceNr / 4;
        neighbors[face1.faceNr] = face0.faceNr / 4;
        ++num;
        ++next;
    }

    if (num > 1)
        atomicAdd(nonManifold, 1);
}

__global__ void markLongEdges(TetDeviceData data, float maxEdgeLength, const float* nodeSizes)
{
    CUDA_THREAD_GUARD(edgeIndex, data.numEdges)
    const std::uint64_t edge = data.edges[edgeIndex];
    const int id0 = static_cast<int>(edge >> 32);
    const int id1 = static_cast<int>(edge & 0xffffffffu);
    const Vec3 delta = data.nodes[id1] - data.nodes[id0];
    float limit = maxEdgeLength;
    if (nodeSizes)
    {
        const float local = fminf(nodeSizes[id0], nodeSizes[id1]);
        limit = local > 0.0f && isfinite(local) ? local : 0.0f;
    }
    data.firstCutVertex[edgeIndex] = limit > 0.0f && delta.magnitudeSquared() > limit * limit ? 1 : 0;
}

__global__ void createMarkedSubdivisionVertices(TetDeviceData data, int originalNodeCount)
{
    CUDA_THREAD_GUARD(edgeIndex, data.numEdges)
    if (data.firstCutVertex[edgeIndex + 1] == data.firstCutVertex[edgeIndex])
    {
        data.edgeCutVertices[edgeIndex] = -1;
        return;
    }

    const std::uint64_t edge = data.edges[edgeIndex];
    const int id0 = static_cast<int>(edge >> 32);
    const int id1 = static_cast<int>(edge & 0xffffffffu);
    const int vertexId = originalNodeCount + data.firstCutVertex[edgeIndex];
    data.nodes[vertexId] = (data.nodes[id0] + data.nodes[id1]) * 0.5f;
    data.edgeCutVertices[edgeIndex] = vertexId;
}

__global__ void markOppositeBoundaryEdges(TetDeviceData data)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int boundaryFaces[4];
    int count = 0;
    for (int face = 0; face < 4; ++face)
    {
        if (data.tetNeighbors[4 * tetIndex + face] < 0)
            boundaryFaces[count++] = face;
    }

    // Face f is opposite local vertex 3-f. Splitting the edge between the two
    // opposite vertices puts those two boundary faces in different child tets.
    for (int i = 0; i < count; ++i)
    {
        for (int j = i + 1; j < count; ++j)
        {
            const int id0 = data.tetIndices[4 * tetIndex + (3 - boundaryFaces[i])];
            const int id1 = data.tetIndices[4 * tetIndex + (3 - boundaryFaces[j])];
            const int edgeIndex = findPackedEdge(data.edges, packEdge(id0, id1));
            if (edgeIndex >= 0)
                atomicExch(data.firstCutVertex.buffer + edgeIndex, 1);
        }
    }
}

// Fills local[0..9]: corners and edge cut verts (-1 if uncut). Returns cut mask.
__device__ int resolveTetCutLocals(const TetDeviceData& data, int tetIndex, int local[10])
{
    for (int i = 0; i < 4; ++i)
        local[i] = data.tetIndices.buffer[4 * tetIndex + i];

    int mask = 0;
    for (int e = 0; e < 6; ++e)
    {
        const int id0 = local[kTetEdges[e][0]];
        const int id1 = local[kTetEdges[e][1]];
        const int edgeIndex = findPackedEdge(data.edges, packEdge(id0, id1));
        const int cutId = edgeIndex >= 0 ? data.edgeCutVertices.buffer[edgeIndex] : -1;
        local[4 + e] = cutId;
        if (cutId >= 0)
            mask |= 1 << e;
    }
    return mask;
}

__device__ int resolveDiagBits(const tet_cut::CutTemplateTables& tables, int mask, const int local[10])
{
    int diagBits = 0;
    for (int f = 0; f < 4; ++f)
    {
        const int a = tables.diagA[mask][f];
        const int b = tables.diagB[mask][f];
        if (a < 0)
            continue;
        if (local[a] > local[b])
            diagBits |= 1 << f;
    }
    return diagBits;
}

void uploadCutTables(TetDeviceData& data)
{
    tet_cut::CutTemplateTables hostTables;
    tet_cut::buildCutTemplateTables(hostTables);
    if (hostTables.childCount[0][0] != 1 || hostTables.childCount[1][0] != 2 ||
        hostTables.childCount[63][0] != 8)
        throw std::runtime_error("Tet cut template tables failed sanity checks");
    for (int mask = 0; mask < 64; ++mask)
    {
        for (int diagBits = 0; diagBits < 16; ++diagBits)
        {
            if (hostTables.childCount[mask][diagBits] > tet_cut::kMaxChildren)
                throw std::runtime_error("Tet cut template exceeds max child count");
        }
    }
    data.cutTables.resize(1, false);
    cudaCheck(cudaMemcpy(data.cutTables.buffer, &hostTables, sizeof(hostTables), cudaMemcpyHostToDevice));
}

__global__ void createSteinerVertices(TetDeviceData data, int originalNodeCount, bool countOnly)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int local[10];
    const int mask = resolveTetCutLocals(data, tetIndex, local);
    const tet_cut::CutTemplateTables& tables = data.cutTables[0];
    const int diagBits = resolveDiagBits(tables, mask, local);
    const bool needs = tables.needsSteiner[mask][diagBits] != 0;

    if (countOnly)
    {
        data.firstSteiner[tetIndex] = needs ? 1 : 0;
        return;
    }

    if (!needs)
    {
        data.steinerVertexId[tetIndex] = -1;
        return;
    }

    const int vertexId = originalNodeCount + data.firstSteiner[tetIndex];
    data.nodes[vertexId] =
        (data.nodes[local[0]] + data.nodes[local[1]] + data.nodes[local[2]] + data.nodes[local[3]]) * 0.25f;
    data.steinerVertexId[tetIndex] = vertexId;
}

__global__ void splitTets(TetDeviceData data, bool countOnly)
{
    CUDA_THREAD_GUARD(tetIndex, data.numTets)

    int local[11];
    const int mask = resolveTetCutLocals(data, tetIndex, local);
    const tet_cut::CutTemplateTables& tables = data.cutTables[0];
    const int diagBits = resolveDiagBits(tables, mask, local);
    const int childCount = tables.childCount[mask][diagBits];

    if (countOnly)
    {
        data.firstNewTet[tetIndex] = childCount > 0 ? childCount - 1 : 0;
        return;
    }

    if (mask == 0 || childCount <= 1)
        return;

    local[10] = data.steinerVertexId[tetIndex];

    const int firstExtra = data.numTets + data.firstNewTet[tetIndex];
    for (int child = 0; child < childCount; ++child)
    {
        const int outTet = child == 0 ? tetIndex : firstExtra + (child - 1);
        for (int corner = 0; corner < 4; ++corner)
        {
            const int localId = tables.children[mask][diagBits][child * 4 + corner];
            data.tetIndices[4 * outTet + corner] = local[localId];
        }
    }
}

void buildUniqueTetEdges(TetDeviceData& data)
{
    if (data.numTets > std::numeric_limits<int>::max() / 6)
        throw std::runtime_error("Tet edge count exceeds the supported range");

    int edgeCount = data.numTets * 6;
    data.edges.resize(static_cast<std::size_t>(edgeCount), false);
    CUDA_LAUNCH(createTetEdges, edgeCount, data);
    sortAndUnique(data.edges, edgeCount);
    data.numEdges = edgeCount;
}

void applyCutTemplates(TetDeviceData& data)
{
    uploadCutTables(data);

    data.firstSteiner.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstSteiner.setZero();
    CUDA_LAUNCH(createSteinerVertices, data.numTets, data, data.numNodes, true);
    thrust::device_ptr<int> firstSteiner(data.firstSteiner.buffer);
    thrust::exclusive_scan(firstSteiner, firstSteiner + data.numTets + 1, firstSteiner);

    const int numSteiner = readDeviceInt(data.firstSteiner, data.numTets);
    data.steinerVertexId.resize(static_cast<std::size_t>(data.numTets), false);
    const int nodesBeforeSteiner = data.numNodes;
    if (numSteiner > 0)
    {
        if (numSteiner > std::numeric_limits<int>::max() - nodesBeforeSteiner)
            throw std::runtime_error("Steiner vertex count exceeds the supported range");
        data.nodes.resize(static_cast<std::size_t>(nodesBeforeSteiner + numSteiner), true);
        data.numNodes += numSteiner;
    }
    CUDA_LAUNCH(createSteinerVertices, data.numTets, data, nodesBeforeSteiner, false);

    data.firstNewTet.resize(static_cast<std::size_t>(data.numTets + 1), false);
    data.firstNewTet.setZero();
    CUDA_LAUNCH(splitTets, data.numTets, data, true);
    thrust::device_ptr<int> firstNewTet(data.firstNewTet.buffer);
    thrust::exclusive_scan(firstNewTet, firstNewTet + data.numTets + 1, firstNewTet);

    const int numExtraTets = readDeviceInt(data.firstNewTet, data.numTets);
    if (numExtraTets > 0)
    {
        if (numExtraTets > std::numeric_limits<int>::max() - data.numTets)
            throw std::runtime_error("Cut tet count exceeds the supported range");
        data.tetIndices.resize(static_cast<std::size_t>(data.numTets + numExtraTets) * 4, true);
    }
    CUDA_LAUNCH(splitTets, data.numTets, data, false);
    data.numTets += numExtraTets;
}

int createMarkedEdgeVertices(TetDeviceData& data)
{
    thrust::device_ptr<int> firstCutVertex(data.firstCutVertex.buffer);
    thrust::exclusive_scan(firstCutVertex, firstCutVertex + data.numEdges + 1, firstCutVertex);

    const int numCutVertices = readDeviceInt(data.firstCutVertex, data.numEdges);
    data.edgeCutVertices.resize(static_cast<std::size_t>(data.numEdges), false);
    const int originalNodeCount = data.numNodes;
    if (numCutVertices > 0)
    {
        if (numCutVertices > std::numeric_limits<int>::max() - originalNodeCount)
            throw std::runtime_error("Cut vertex count exceeds the supported range");
        data.nodes.resize(static_cast<std::size_t>(originalNodeCount + numCutVertices), true);
        data.numNodes += numCutVertices;
    }
    CUDA_LAUNCH(createMarkedSubdivisionVertices, data.numEdges, data, originalNodeCount);

    if (numCutVertices > 0)
        applyCutTemplates(data);
    return numCutVertices;
}

} // namespace

bool computeNeighbors(TetDeviceData& data)
{
    if (data.numTets <= 0)
    {
        data.tetNeighbors.free();
        return true;
    }

    if (data.numTets > std::numeric_limits<int>::max() / 4)
        throw std::runtime_error("Tet face count exceeds the supported range");

    const int numFaces = data.numTets * 4;
    data.tetNeighbors.resize(static_cast<std::size_t>(numFaces), false);

    DeviceBuffer<Face> faces;
    faces.resize(static_cast<std::size_t>(numFaces), false);
    CUDA_LAUNCH(fillFaces, data.numTets, data, faces.buffer);

    thrust::device_ptr<Face> facePtr(faces.buffer);
    thrust::sort(facePtr, facePtr + numFaces, FaceComparator());

    DeviceBuffer<int> nonManifold;
    nonManifold.resize(1, false);
    nonManifold.setZero();
    CUDA_LAUNCH(fillNeighbors, numFaces, numFaces, faces.buffer, data.tetNeighbors.buffer, nonManifold.buffer);

    const bool manifold = readDeviceInt(nonManifold, 0) == 0;
    faces.free();
    nonManifold.free();
    return manifold;
}

int subdivideLongEdges(TetDeviceData& data, float maxEdgeLength, const float* nodeSizes)
{
    if (!nodeSizes && !(maxEdgeLength > 0.0f))
        return 0;

    buildUniqueTetEdges(data);

    data.firstCutVertex.resize(static_cast<std::size_t>(data.numEdges + 1), false);
    data.firstCutVertex.setZero();
    CUDA_LAUNCH(markLongEdges, data.numEdges, data, maxEdgeLength, nodeSizes);
    return createMarkedEdgeVertices(data);
}

void separateBoundaryFaces(TetDeviceData& data)
{
    constexpr int kMaxPasses = 32;
    for (int pass = 0; pass < kMaxPasses; ++pass)
    {
        buildUniqueTetEdges(data);
        data.firstCutVertex.resize(static_cast<std::size_t>(data.numEdges + 1), false);
        data.firstCutVertex.setZero();
        CUDA_LAUNCH(markOppositeBoundaryEdges, data.numTets, data);
        const int numEdgeSplits = createMarkedEdgeVertices(data);
        if (numEdgeSplits == 0)
            return;
        if (!computeNeighbors(data))
            throw std::runtime_error("Tet mesh has non-manifold faces after boundary edge splitting");
    }
    throw std::runtime_error("Boundary face separation did not converge");
}

} // namespace tetrahedralizer
