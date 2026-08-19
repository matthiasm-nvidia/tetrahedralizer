#pragma once

struct ImFont;
struct GLFWwindow;

namespace viewer::imgui_theme
{
void apply_kit_dark_style();
bool load_ui_font(GLFWwindow* window);
ImFont* ui_bold_font();

} // namespace viewer::imgui_theme
