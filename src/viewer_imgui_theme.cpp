#include "viewer_imgui_theme.h"

#define NOMINMAX

#ifdef _WIN32
#    include <windows.h>
#endif

#include "imgui.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <filesystem>
#include <string>

namespace viewer::imgui_theme
{
namespace
{
std::string g_font_path;
ImFont* g_bold_font = nullptr;

std::string try_font_candidate(const std::filesystem::path& candidate)
{
    if (std::filesystem::exists(candidate))
    {
        return candidate.string();
    }
    return {};
}

std::string find_font_near(const std::filesystem::path& start_dir)
{
    std::filesystem::path dir = start_dir;
    for (int depth = 0; depth < 8 && !dir.empty(); ++depth, dir = dir.parent_path())
    {
        if (const std::string found = try_font_candidate(dir / "resources" / "fonts" / "NotoSans-Regular.ttf");
            !found.empty())
        {
            return found;
        }
    }
    return {};
}

float content_scale(GLFWwindow* window)
{
    if (window == nullptr)
    {
        return 1.0f;
    }

    int window_w = 0;
    int window_h = 0;
    int framebuffer_w = 0;
    int framebuffer_h = 0;
    glfwGetWindowSize(window, &window_w, &window_h);
    glfwGetFramebufferSize(window, &framebuffer_w, &framebuffer_h);
    if (window_w <= 0)
    {
        return 1.0f;
    }
    return static_cast<float>(framebuffer_w) / static_cast<float>(window_w);
}
} // namespace

ImFont* ui_bold_font()
{
    return g_bold_font;
}

void apply_kit_dark_style()
{
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowPadding = ImVec2(10.0f, 10.0f);
    style.WindowRounding = 2.0f;
    style.ChildWindowRounding = 0.0f;
    style.FramePadding = ImVec2(8.0f, 4.0f);
    style.FrameRounding = 3.0f;
    style.ItemSpacing = ImVec2(8.0f, 10.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 21.0f;
    style.ScrollbarSize = 16.0f;
    style.GrabMinSize = 12.0f;
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.ButtonTextAlign = ImVec2(0.48f, 0.5f);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = ImVec4(0.878f, 0.878f, 0.878f, 1.0f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.294f, 0.294f, 0.294f, 1.0f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.150f, 0.150f, 0.150f, 1.0f);
    colors[ImGuiCol_ChildWindowBg] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_Border] = ImVec4(0.294f, 0.294f, 0.294f, 1.0f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.086f, 0.086f, 0.086f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.086f, 0.086f, 0.086f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.192f, 0.192f, 0.192f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.247f, 0.247f, 0.247f, 1.0f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.086f, 0.086f, 0.086f, 1.0f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.388f, 0.388f, 0.388f, 1.0f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.490f, 0.490f, 0.490f, 1.0f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.541f, 0.541f, 0.541f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.149f, 0.149f, 0.149f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.388f, 0.388f, 0.388f, 1.0f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.490f, 0.490f, 0.490f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.133f, 0.133f, 0.133f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.192f, 0.192f, 0.192f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.0f, 0.376f, 0.780f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.133f, 0.133f, 0.133f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.192f, 0.192f, 0.192f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.220f, 0.220f, 0.220f, 1.0f);
    colors[ImGuiCol_Separator] = ImVec4(0.294f, 0.294f, 0.294f, 1.0f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.294f, 0.294f, 0.294f, 1.0f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.294f, 0.294f, 0.294f, 1.0f);
}

bool load_ui_font(GLFWwindow* window)
{
#ifdef _WIN32
    if (wchar_t module_path[MAX_PATH]; GetModuleFileNameW(nullptr, module_path, MAX_PATH) != 0)
    {
        g_font_path = find_font_near(std::filesystem::path(module_path).parent_path());
    }
#endif
    if (g_font_path.empty())
    {
        g_font_path = find_font_near(std::filesystem::current_path());
    }
    if (g_font_path.empty())
    {
        std::fprintf(stderr, "reconstructor: UI font not found (expected resources/fonts/NotoSans-Regular.ttf)\n");
        return false;
    }

    const float dpi_scale = content_scale(window);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    g_bold_font = nullptr;
    const float font_size = 22.0f * dpi_scale;
    ImFont* font = io.Fonts->AddFontFromFileTTF(g_font_path.c_str(), font_size);
    if (font == nullptr)
    {
        std::fprintf(stderr, "reconstructor: failed to load UI font from %s\n", g_font_path.c_str());
        g_font_path.clear();
        return false;
    }

    const std::filesystem::path bold_font_path = std::filesystem::path(g_font_path).parent_path() / "NotoSans-Bold.ttf";
    if (std::filesystem::exists(bold_font_path))
    {
        g_bold_font = io.Fonts->AddFontFromFileTTF(bold_font_path.string().c_str(), font_size);
        if (g_bold_font == nullptr)
        {
            std::fprintf(
                stderr, "reconstructor: failed to load bold UI font from %s\n", bold_font_path.string().c_str());
        }
    }

#ifndef __APPLE__
    if (dpi_scale > 1.0f)
    {
        ImGui::GetStyle().ScaleAllSizes(dpi_scale);
    }
#endif

    std::fprintf(stdout, "reconstructor: loaded UI font %s (size=%.1f, dpiScale=%.2f)\n", g_font_path.c_str(),
                 font_size, dpi_scale);
    return true;
}

} // namespace viewer::imgui_theme
