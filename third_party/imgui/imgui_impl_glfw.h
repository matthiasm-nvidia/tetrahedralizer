
// ImGui GLFW binding with OpenGL
// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplGLFW_Init(), ImGui_ImplGLFW_NewFrame(),
// ImGui::Render() and ImGui_ImplGLFW_Shutdown(). If you are new to ImGui, see examples/README.txt and documentation at
// the top of imgui.cpp. https://github.com/ocornut/imgui

#pragma once

#include "imgui.h"

struct GLFWwindow;

IMGUI_API bool ImGui_ImplGLFW_Init(GLFWwindow* window, bool install_callbacks);
IMGUI_API void ImGui_ImplGLFW_Shutdown();
IMGUI_API void ImGui_ImplGLFW_NewFrame();
IMGUI_API void ImGui_ImplGLFW_RenderDrawData(ImDrawData* draw_data);

// Use if you want to reset your rendering device without losing ImGui state.
IMGUI_API void ImGui_ImplGLFW_InvalidateDeviceObjects();
IMGUI_API bool ImGui_ImplGLFW_CreateDeviceObjects();

// GLFW callbacks (installed by default if you enable 'install_callbacks' during initialization)
// You can also handle inputs yourself and use those as a reference.
IMGUI_API void ImGui_ImplGLFW_MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
IMGUI_API void ImGui_ImplGLFW_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
IMGUI_API void ImGui_ImplGLFW_KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
IMGUI_API void ImGui_ImplGLFW_CharCallback(GLFWwindow* window, unsigned int c);
