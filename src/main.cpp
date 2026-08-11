#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "FileDialog.h"
#include "tetrahedralizer/Camera.h"
#include "tetrahedralizer/GlCore.h"
#include "tetrahedralizer/LastPath.h"
#include "tetrahedralizer/TriMesh.h"
#include "tetrahedralizer/TriMeshRenderer.h"
#include "viewer_imgui_theme.h"
#include "viewer_imgui_widgets.h"

namespace
{
constexpr float kBaseCameraSpeed = 0.075f;

struct AppState
{
    tetrahedralizer::Camera camera;
    tetrahedralizer::Vec3 orbit_center{0.0f, 0.0f, 0.0f};
    float scene_scale = 1.0f;
    int mouse_x = 0;
    int mouse_y = 0;
};

AppState* appState(GLFWwindow* window)
{
    return static_cast<AppState*>(glfwGetWindowUserPointer(window));
}

bool imguiWantsKeyboard()
{
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
}

bool imguiWantsMouse()
{
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
}

bool loadMesh(GLFWwindow* window, const std::string& path, AppState& state,
              tetrahedralizer::TriMesh& mesh, tetrahedralizer::TriMeshRenderer& renderer)
{
    if (!std::filesystem::exists(path))
        return false;

    tetrahedralizer::TriMesh loaded_mesh;
    if (!loaded_mesh.loadObj(path.c_str()))
        return false;

    mesh = std::move(loaded_mesh);
    renderer.upload(mesh.positions, mesh.triangle_indices);

    const tetrahedralizer::Bounds3 bounds = mesh.bounds();
    tetrahedralizer::frameCamera(state.camera, state.orbit_center, bounds);
    const float diagonal = bounds.getDimensions().length();
    state.scene_scale = diagonal > 0.0f ? diagonal : 1.0f;
    state.camera.speed = kBaseCameraSpeed * state.scene_scale;

    tetrahedralizer::writeLastPath(path);
    glfwSetWindowTitle(window, ("Tetrahedralizer - " + path).c_str());
    return true;
}

void promptLoadMesh(GLFWwindow* window, AppState& state, tetrahedralizer::TriMesh& mesh,
                    tetrahedralizer::TriMeshRenderer& renderer)
{
    std::string path;
    if (FileDialog::getFileName(path, true) && !loadMesh(window, path, state, mesh, renderer))
        FileDialog::showError("Load Failed", "Failed to load OBJ mesh.");
}

void loadLastMesh(GLFWwindow* window, AppState& state, tetrahedralizer::TriMesh& mesh,
                  tetrahedralizer::TriMeshRenderer& renderer)
{
    std::string path;
    if (!tetrahedralizer::readLastPath(path) || !loadMesh(window, path, state, mesh, renderer))
        promptLoadMesh(window, state, mesh, renderer);
}

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    ImGui_ImplGLFW_KeyCallback(window, key, scancode, action, mods);
    AppState* state = appState(window);
    if (state == nullptr || imguiWantsKeyboard() || (action != GLFW_PRESS && action != GLFW_REPEAT))
        return;

    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
        state->camera.speed *= 2.0f;
    else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
        state->camera.speed *= 0.5f;
    else
        state->camera.handle_key_down(key);
}

void onMouseButton(GLFWwindow* window, int button, int action, int mods)
{
    ImGui_ImplGLFW_MouseButtonCallback(window, button, action, mods);

    AppState* state = appState(window);
    if (state == nullptr)
        return;

    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    state->mouse_x = static_cast<int>(x);
    state->mouse_y = static_cast<int>(y);
}

void onMouseScroll(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGui_ImplGLFW_ScrollCallback(window, xoffset, yoffset);
    AppState* state = appState(window);
    if (state != nullptr && !imguiWantsMouse())
        state->camera.handle_wheel(static_cast<int>(yoffset));
}

void onMouseMotion(GLFWwindow* window, double x, double y)
{
    AppState* state = appState(window);
    if (state == nullptr)
        return;

    const int mouse_x = static_cast<int>(x);
    const int mouse_y = static_cast<int>(y);
    const int dx = mouse_x - state->mouse_x;
    const int dy = mouse_y - state->mouse_y;
    state->mouse_x = mouse_x;
    state->mouse_y = mouse_y;

    if (imguiWantsMouse())
        return;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        state->camera.handle_mouse_view(dx, dy);
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
    {
        const tetrahedralizer::Vec3 old_position = state->camera.pos;
        const float scale = (state->camera.pos - state->orbit_center).magnitude() * 0.001f;
        state->camera.handle_mouse_translate(dx, dy, scale);
        state->orbit_center += state->camera.pos - old_position;
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        state->camera.handle_mouse_orbit(dx, dy, state->orbit_center);
}

void installCallbacks(GLFWwindow* window, AppState* state)
{
    glfwSetWindowUserPointer(window, state);
    glfwSetKeyCallback(window, onKey);
    glfwSetCharCallback(window, ImGui_ImplGLFW_CharCallback);
    glfwSetMouseButtonCallback(window, onMouseButton);
    glfwSetScrollCallback(window, onMouseScroll);
    glfwSetCursorPosCallback(window, onMouseMotion);
}

} // namespace

int main()
{
    if (!glfwInit())
    {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1600, 900, "Tetrahedralizer", nullptr, nullptr);
    if (window == nullptr)
    {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!tetrahedralizer::glLoad())
    {
        std::fprintf(stderr, "Failed to load OpenGL functions\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    ImGuiContext* imgui_context = ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    viewer::imgui_theme::apply_kit_dark_style();
    viewer::imgui_theme::load_ui_font(window);
    ImGui_ImplGLFW_Init(window, false);

    AppState state;
    state.camera.init();
    state.camera.speed = kBaseCameraSpeed;
    installCallbacks(window, &state);

    tetrahedralizer::TriMesh mesh;
    tetrahedralizer::TriMeshRenderer renderer;
    bool show_mesh = true;
    bool wireframe = false;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        state.camera.applyProjection(width, height, 100000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        state.camera.applyView();
        if (show_mesh)
            renderer.render(wireframe);

        ImGui_ImplGLFW_NewFrame();
        ImGui::SetNextWindowSize(ImVec2(320.0f, 300.0f), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Controls");
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float half_width = (ImGui::GetContentRegionAvail().x - spacing) * 0.5f;
        if (viewer::imgui_widgets::button("Load OBJ", half_width))
            promptLoadMesh(window, state, mesh, renderer);
        ImGui::SameLine();
        if (viewer::imgui_widgets::button("Last", half_width))
            loadLastMesh(window, state, mesh, renderer);

        if (!mesh.empty())
        {
            viewer::imgui_widgets::section_separator();
            viewer::imgui_widgets::section_heading("Mesh");
            ImGui::Text("%llu vertices", static_cast<unsigned long long>(mesh.positions.size()));
            ImGui::Text("%llu triangles",
                        static_cast<unsigned long long>(mesh.triangle_indices.size() / 3));
            viewer::imgui_widgets::checkbox("Show mesh", &show_mesh);
            viewer::imgui_widgets::checkbox("Wireframe", &wireframe);
        }

        viewer::imgui_widgets::section_separator();
        ImGui::TextWrapped("Left drag: orbit | Right drag: look");
        ImGui::TextWrapped("Middle drag: pan | Wheel/WASDQE: move");
        ImGui::End();
        ImGui::Render();
        ImGui_ImplGLFW_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    renderer.clear();
    ImGui_ImplGLFW_Shutdown();
    ImGui::DestroyContext(imgui_context);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
