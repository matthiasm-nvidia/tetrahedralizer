#include "FileDialog.h"

#ifdef _WIN32

#    include <shobjidl.h>
#    include <vector>
#    include <windows.h>

bool FileDialog::getFileName(std::string& filename, bool open)
{
    bool OK = false;

    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr))
    {
        IFileOpenDialog* pFileOpenDialog = nullptr;
        IFileSaveDialog* pFileSaveDialog = nullptr;

        if (open)
        {
            hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog,
                                  reinterpret_cast<void**>(&pFileOpenDialog));
        }
        else
        {
            hr = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_ALL, IID_IFileSaveDialog,
                                  reinterpret_cast<void**>(&pFileSaveDialog));
        }

        if (SUCCEEDED(hr))
        {
            COMDLG_FILTERSPEC rgSpec[] = {
                {L"Wavefront OBJ Meshes", L"*.obj"},
                {L"All Files", L"*.*"},
            };

            if (open)
            {
                pFileOpenDialog->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
                hr = pFileOpenDialog->Show(NULL);
            }
            else
            {
                pFileSaveDialog->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
                hr = pFileSaveDialog->Show(NULL);
            }

            if (SUCCEEDED(hr))
            {
                IShellItem* pItem;
                if (open)
                    hr = pFileOpenDialog->GetResult(&pItem);
                else
                    hr = pFileSaveDialog->GetResult(&pItem);

                if (SUCCEEDED(hr))
                {
                    PWSTR pszFilePath;
                    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                    if (SUCCEEDED(hr))
                    {
                        size_t convertedChars = 0;
                        size_t wideStrSize = wcslen(pszFilePath) + 1;

                        std::vector<char> cStr(wideStrSize * sizeof(wchar_t));

                        convertedChars = wcstombs(&cStr[0], pszFilePath, wideStrSize);
                        if (convertedChars == static_cast<size_t>(-1))
                            convertedChars = 0;
                        cStr[convertedChars] = '\0';

                        filename = std::string(&cStr[0]);
                        OK = true;

                        CoTaskMemFree(pszFilePath);
                    }
                    pItem->Release();
                }
            }

            if (open)
                pFileOpenDialog->Release();
            else
                pFileSaveDialog->Release();
        }
        CoUninitialize();
    }
    return OK;
}

void FileDialog::showError(const std::string& title, const std::string& message)
{
    MessageBoxA(nullptr, message.c_str(), title.c_str(), MB_OK | MB_ICONERROR);
}

#else

#    include <cstdio>

bool FileDialog::getFileName(std::string& filename, bool open)
{
    std::string command;
    if (open)
    {
        command = "zenity --file-selection --title='Open OBJ Mesh' "
                  "--file-filter='Wavefront OBJ Meshes | *.obj' "
                  "--file-filter='All Files | *'";
    }
    else
    {
        command = "zenity --file-selection --save --title='Save OBJ Mesh' "
                  "--file-filter='Wavefront OBJ Meshes | *.obj' "
                  "--file-filter='All Files | *'";
    }

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
        return false;

    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        result += buffer;

    const int status = pclose(pipe);

    if (!result.empty() && result.back() == '\n')
        result.pop_back();

    if (status == 0 && !result.empty())
    {
        filename = result;
        return true;
    }

    return false;
}

void FileDialog::showError(const std::string& title, const std::string& message)
{
    std::string command = "zenity --error --title='" + title + "' --text='" + message + "'";
    (void)system(command.c_str());
}

#endif
