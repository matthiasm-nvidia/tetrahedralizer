#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace tetrahedralizer
{
namespace tet_cut
{

// Edge / face layout matches D:\GIT\tet-cut (Ruprecht–Müller-style tables).
// bit0=01, bit1=12, bit2=20, bit3=03, bit4=13, bit5=23
constexpr int kTetEdges[6][2] = {
    {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3},
};
constexpr int kTetFaces[4][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3},
};

constexpr int kMaxChildren = 12;
constexpr int kMaxChildCorners = kMaxChildren * 4;

struct CutTemplateTables
{
    int childCount[64][16] = {};
    int needsSteiner[64][16] = {};
    int diagA[64][4] = {}; // -1 if face has no diagonal choice
    int diagB[64][4] = {};
    int children[64][16][kMaxChildCorners] = {};
};

inline int edgeIndexOf(int p, int q)
{
    for (int e = 0; e < 6; ++e)
    {
        const int a = kTetEdges[e][0];
        const int b = kTetEdges[e][1];
        if ((a == p && b == q) || (a == q && b == p))
            return e;
    }
    return -1;
}

inline float refVolume(int ia, int ib, int ic, int id)
{
    // Reference tet + edge midpoints + centroid (local 0..10).
    static constexpr float kRef[11][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {0.5f, 0, 0}, {0.5f, 0.5f, 0}, {0, 0.5f, 0},
        {0, 0, 0.5f}, {0.5f, 0, 0.5f}, {0, 0.5f, 0.5f},
        {0.25f, 0.25f, 0.25f},
    };
    const float* a = kRef[ia];
    const float bx = kRef[ib][0] - a[0], by = kRef[ib][1] - a[1], bz = kRef[ib][2] - a[2];
    const float cx = kRef[ic][0] - a[0], cy = kRef[ic][1] - a[1], cz = kRef[ic][2] - a[2];
    const float dx = kRef[id][0] - a[0], dy = kRef[id][1] - a[1], dz = kRef[id][2] - a[2];
    return ((by * cz - bz * cy) * dx + (bz * cx - bx * cz) * dy + (bx * cy - by * cx) * dz) / 6.0f;
}

// Only a face with exactly two cut edges leaves a quad, so only then is there a
// diagonal to pick. Its endpoints are the ends of the uncut edge.
inline void faceDiagPair(int a, int b, int c, int mask, int& diagA, int& diagB)
{
    auto midOf = [mask](int p, int q) -> int {
        const int e = edgeIndexOf(p, q);
        return ((mask >> e) & 1) ? 4 + e : -1;
    };

    const int mab = midOf(a, b);
    const int mbc = midOf(b, c);
    const int mca = midOf(c, a);
    const int n = (mab >= 0 ? 1 : 0) + (mbc >= 0 ? 1 : 0) + (mca >= 0 ? 1 : 0);

    diagA = -1;
    diagB = -1;
    if (n != 2)
        return;

    if (mca < 0)
    {
        diagA = a;
        diagB = c;
    }
    else if (mab < 0)
    {
        diagA = b;
        diagB = a;
    }
    else
    {
        diagA = c;
        diagB = b;
    }
}

inline void faceTrisForDiag(int a, int b, int c, int mask, int diagBit, std::vector<std::array<int, 3>>& out)
{
    auto midOf = [mask](int p, int q) -> int {
        const int e = edgeIndexOf(p, q);
        return ((mask >> e) & 1) ? 4 + e : -1;
    };

    const int mab = midOf(a, b);
    const int mbc = midOf(b, c);
    const int mca = midOf(c, a);
    const int n = (mab >= 0 ? 1 : 0) + (mbc >= 0 ? 1 : 0) + (mca >= 0 ? 1 : 0);

    if (n == 0)
    {
        out.push_back({a, b, c});
        return;
    }
    if (n == 3)
    {
        out.push_back({a, mab, mca});
        out.push_back({mab, b, mbc});
        out.push_back({mca, mbc, c});
        out.push_back({mab, mbc, mca});
        return;
    }
    if (n == 1)
    {
        if (mab >= 0)
        {
            out.push_back({a, mab, c});
            out.push_back({mab, b, c});
        }
        else if (mbc >= 0)
        {
            out.push_back({b, mbc, a});
            out.push_back({mbc, c, a});
        }
        else
        {
            out.push_back({c, mca, b});
            out.push_back({mca, a, b});
        }
        return;
    }

    int A, B, C, mAB, mBC;
    if (mca < 0)
    {
        A = a;
        B = b;
        C = c;
        mAB = mab;
        mBC = mbc;
    }
    else if (mab < 0)
    {
        A = b;
        B = c;
        C = a;
        mAB = mbc;
        mBC = mca;
    }
    else
    {
        A = c;
        B = a;
        C = b;
        mAB = mca;
        mBC = mab;
    }

    out.push_back({mAB, B, mBC});
    if (diagBit == 0)
    {
        out.push_back({A, mAB, mBC});
        out.push_back({A, mBC, C});
    }
    else
    {
        out.push_back({A, mAB, C});
        out.push_back({mAB, mBC, C});
    }
}

inline void buildTemplate(int mask, int diagBits, CutTemplateTables& tables)
{
    if (mask == 0)
    {
        tables.childCount[mask][diagBits] = 1;
        tables.needsSteiner[mask][diagBits] = 0;
        tables.children[mask][diagBits][0] = 0;
        tables.children[mask][diagBits][1] = 1;
        tables.children[mask][diagBits][2] = 2;
        tables.children[mask][diagBits][3] = 3;
        return;
    }

    auto midLocal = [mask](int p, int q) -> int {
        const int e = edgeIndexOf(p, q);
        return ((mask >> e) & 1) ? 4 + e : -1;
    };

    std::vector<std::array<int, 3>> tris;
    for (int f = 0; f < 4; ++f)
        faceTrisForDiag(kTetFaces[f][0], kTetFaces[f][1], kTetFaces[f][2], mask, (diagBits >> f) & 1, tris);

    std::vector<std::array<int, 4>> children;

    for (int v = 0; v < 4; ++v)
    {
        int ms[3];
        int mi = 0;
        bool allCut = true;
        for (int u = 0; u < 4; ++u)
        {
            if (u == v)
                continue;
            ms[mi] = midLocal(v, u);
            if (ms[mi] < 0)
                allCut = false;
            ++mi;
        }
        if (!allCut)
            continue;

        std::vector<std::array<int, 3>> kept;
        for (const auto& t : tris)
        {
            const bool cornerTri = (t[0] == v || t[1] == v || t[2] == v) &&
                                   ((t[0] == v || t[0] == ms[0] || t[0] == ms[1] || t[0] == ms[2]) &&
                                    (t[1] == v || t[1] == ms[0] || t[1] == ms[1] || t[1] == ms[2]) &&
                                    (t[2] == v || t[2] == ms[0] || t[2] == ms[1] || t[2] == ms[2]));
            if (!cornerTri)
                kept.push_back(t);
        }
        tris.swap(kept);
        children.push_back({v, ms[0], ms[1], ms[2]});
        tris.push_back({ms[0], ms[1], ms[2]});
    }

    std::vector<int> present;
    for (const auto& t : tris)
        for (int x : t)
            present.push_back(x);
    std::sort(present.begin(), present.end());
    present.erase(std::unique(present.begin(), present.end()), present.end());

    int apex = -1;
    int fewest = 0x7fffffff;
    for (int v : present)
    {
        int count = 0;
        bool ok = true;
        for (const auto& t : tris)
        {
            if (t[0] == v || t[1] == v || t[2] == v)
                continue;
            if (std::fabs(refVolume(v, t[0], t[1], t[2])) < 1.0e-9f)
            {
                ok = false;
                break;
            }
            ++count;
        }
        if (ok && count < fewest)
        {
            fewest = count;
            apex = v;
        }
    }
    if (apex < 0)
        apex = 10;

    for (const auto& t : tris)
    {
        if (t[0] == apex || t[1] == apex || t[2] == apex)
            continue;
        children.push_back({apex, t[0], t[1], t[2]});
    }

    int needsSteiner = 0;
    const int childCount = static_cast<int>(children.size());
    for (int i = 0; i < childCount; ++i)
    {
        auto ch = children[i];
        if (ch[0] == 10 || ch[1] == 10 || ch[2] == 10 || ch[3] == 10)
            needsSteiner = 1;
        if (refVolume(ch[0], ch[1], ch[2], ch[3]) < 0.0f)
            std::swap(ch[0], ch[1]);
        tables.children[mask][diagBits][i * 4 + 0] = ch[0];
        tables.children[mask][diagBits][i * 4 + 1] = ch[1];
        tables.children[mask][diagBits][i * 4 + 2] = ch[2];
        tables.children[mask][diagBits][i * 4 + 3] = ch[3];
    }
    tables.childCount[mask][diagBits] = childCount;
    tables.needsSteiner[mask][diagBits] = needsSteiner;
}

inline void buildCutTemplateTables(CutTemplateTables& tables)
{
    for (int mask = 0; mask < 64; ++mask)
    {
        for (int f = 0; f < 4; ++f)
        {
            faceDiagPair(kTetFaces[f][0], kTetFaces[f][1], kTetFaces[f][2], mask, tables.diagA[mask][f],
                         tables.diagB[mask][f]);
        }
        for (int diagBits = 0; diagBits < 16; ++diagBits)
            buildTemplate(mask, diagBits, tables);
    }
}

} // namespace tet_cut
} // namespace tetrahedralizer
