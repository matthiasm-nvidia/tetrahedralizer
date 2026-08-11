#include "tetrahedralizer/LastPath.h"

#include <fstream>
#include <string>

#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>
#else
#    include <climits>
#    include <unistd.h>
#endif

namespace tetrahedralizer
{
namespace
{
void trimInPlace(std::string& text)
{
    const size_t start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
    {
        text.clear();
        return;
    }

    const size_t end = text.find_last_not_of(" \t\r\n");
    text = text.substr(start, end - start + 1);
}
} // namespace

std::string lastPathFile()
{
#ifdef _WIN32
    char modulePath[MAX_PATH];
    const DWORD length = GetModuleFileNameA(nullptr, modulePath, MAX_PATH);
    if (length == 0 || length >= MAX_PATH)
        return "last_mesh.txt";

    std::string path(modulePath, modulePath + length);
    const size_t slash = path.find_last_of("\\/");
    if (slash == std::string::npos)
        return "last_mesh.txt";
    return path.substr(0, slash + 1) + "last_mesh.txt";
#elif defined(__linux__)
    char modulePath[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", modulePath, sizeof(modulePath) - 1);
    if (length <= 0)
        return "last_mesh.txt";

    modulePath[length] = '\0';
    std::string path(modulePath, modulePath + length);
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos)
        return "last_mesh.txt";
    return path.substr(0, slash + 1) + "last_mesh.txt";
#else
    return "last_mesh.txt";
#endif
}

bool readLastPath(std::string& path)
{
    std::ifstream file(lastPathFile());
    if (!file)
        return false;

    std::string line;
    if (!std::getline(file, line))
        return false;

    trimInPlace(line);
    if (line.empty())
        return false;

    path = std::move(line);
    return true;
}

void writeLastPath(const std::string& path)
{
    if (path.empty())
        return;

    std::ofstream file(lastPathFile(), std::ios::trunc);
    if (!file)
        return;

    file << path;
}

} // namespace tetrahedralizer
