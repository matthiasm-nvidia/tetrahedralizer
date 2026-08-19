#pragma once

#include <string>

namespace tetrahedralizer
{

std::string lastPathFile();
bool readLastPath(std::string& path);
void writeLastPath(const std::string& path);

} // namespace tetrahedralizer
