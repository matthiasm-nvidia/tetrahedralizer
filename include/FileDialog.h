#pragma once

#include <string>

// Cross-platform file dialog
// Windows: native COM IFileOpenDialog
// Linux: zenity
class FileDialog
{
public:
    static bool getFileName(std::string& filename, bool open);
    static void showError(const std::string& title, const std::string& message);
};
