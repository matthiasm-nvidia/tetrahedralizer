#include "tetrahedralizer/TriMesh.h"

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace tetrahedralizer
{
namespace
{

bool parseVertexIndex(const std::string& token, std::size_t vertex_count, std::uint32_t& index)
{
    const std::size_t slash = token.find('/');
    const std::string index_text = token.substr(0, slash);
    if (index_text.empty())
        return false;

    std::size_t parsed = 0;
    long long obj_index = 0;
    try
    {
        obj_index = std::stoll(index_text, &parsed);
    }
    catch (...)
    {
        return false;
    }
    if (parsed != index_text.size() || obj_index == 0)
        return false;

    const long long resolved =
        obj_index > 0 ? obj_index - 1 : static_cast<long long>(vertex_count) + obj_index;
    if (resolved < 0 || resolved >= static_cast<long long>(vertex_count))
        return false;

    index = static_cast<std::uint32_t>(resolved);
    return true;
}

} // namespace

bool TriMesh::loadObj(const char* path)
{
    std::ifstream input(path);
    if (!input)
        return false;

    std::vector<Vec3> loaded_positions;
    std::vector<std::uint32_t> loaded_indices;
    std::string line;
    while (std::getline(input, line))
    {
        std::istringstream stream(line);
        std::string type;
        stream >> type;
        if (type.empty() || type[0] == '#')
            continue;

        if (type == "v")
        {
            Vec3 position;
            if (!(stream >> position.x >> position.y >> position.z))
                return false;
            loaded_positions.push_back(position);
        }
        else if (type == "f")
        {
            std::vector<std::uint32_t> face;
            std::string token;
            while (stream >> token)
            {
                if (!token.empty() && token[0] == '#')
                    break;

                std::uint32_t index = 0;
                if (!parseVertexIndex(token, loaded_positions.size(), index))
                    return false;
                face.push_back(index);
            }
            if (face.size() < 3)
                return false;

            for (std::size_t i = 1; i + 1 < face.size(); ++i)
            {
                loaded_indices.push_back(face[0]);
                loaded_indices.push_back(face[i]);
                loaded_indices.push_back(face[i + 1]);
            }
        }
    }

    if (loaded_positions.empty() || loaded_indices.empty())
        return false;

    positions = std::move(loaded_positions);
    triangle_indices = std::move(loaded_indices);
    return true;
}

Bounds3 TriMesh::bounds() const
{
    Bounds3 result(Empty);
    for (const Vec3& position : positions)
        result.include(position);
    return result;
}

} // namespace tetrahedralizer
