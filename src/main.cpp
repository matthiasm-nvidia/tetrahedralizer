#include <cstdio>
#include <exception>
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
#include "tetrahedralizer/TetMeshRenderer.h"
#include "tetrahedralizer/Tetrahedralizer.h"
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
    tetrahedralizer::Bounds3 mesh_bounds;
    float scene_scale = 1.0f;
    float voxel_spacing = 0.1f;
    int hole_close_radius = 0;
    int smoothing_iterations = 0;
    float volume_factor = 0.8f;
    bool cut_with_input_mesh = false;
    // Percentage of the mesh bounds cut away along each axis, 0 disables the clip plane.
    int clip[3] = {0, 0, 0};
    int mouse_x = 0;
    int mouse_y = 0;
};

// World space upper bound per axis, MaxFloat where the axis is not clipped.
tetrahedralizer::Vec3 clipLimits(const AppState& state)
{
    const tetrahedralizer::Vec3 dimensions = state.mesh_bounds.getDimensions();
    tetrahedralizer::Vec3 limits(MaxFloat, MaxFloat, MaxFloat);
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        if (state.clip[axis] > 0)
            limits[axis] = state.mesh_bounds.minimum[axis] +
                           dimensions[axis] * (1.0f - static_cast<float>(state.clip[axis]) * 0.01f);
    }
    return limits;
}

void applyClipPlanes(const tetrahedralizer::Vec3& limits)
{
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        if (limits[axis] == MaxFloat)
            continue;

        GLdouble equation[4] = {0.0, 0.0, 0.0, static_cast<GLdouble>(limits[axis])};
        equation[axis] = -1.0;
        glEnable(GL_CLIP_PLANE0 + axis);
        glClipPlane(GL_CLIP_PLANE0 + axis, equation);
    }
}

void disableClipPlanes()
{
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
}

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

void createTets(const tetrahedralizer::TriMesh& mesh, float voxel_spacing, int hole_close_radius,
                int smoothing_iterations, float volume_factor, bool cut_with_input_mesh,
                tetrahedralizer::Tetrahedralizer& tets, tetrahedralizer::TetMeshRenderer& tet_renderer)
{
    try
    {
        tetrahedralizer::TetrahedralizerParams params;
        params.voxelSpacing = voxel_spacing;
        params.holeCloseRadius = hole_close_radius;
        params.numSmoothingIterations = smoothing_iterations;
        params.volumeFactor = volume_factor;
        params.cutWithInputMesh = cut_with_input_mesh;
        tets.create(mesh.positions, mesh.triangle_indices, params);
        tet_renderer.upload(tets.nodes, tets.tet_indices);
    }
    catch (const std::exception& error)
    {
        tets.clear();
        tet_renderer.clear();
        FileDialog::showError("Tetrahedralization Failed", error.what());
    }
}

bool loadMesh(GLFWwindow* window, const std::string& path, AppState& state, tetrahedralizer::TriMesh& mesh,
              tetrahedralizer::TriMeshRenderer& tri_renderer, tetrahedralizer::Tetrahedralizer& tets,
              tetrahedralizer::TetMeshRenderer& tet_renderer)
{
    if (!std::filesystem::exists(path))
        return false;

    tetrahedralizer::TriMesh loaded_mesh;
    if (!loaded_mesh.loadObj(path.c_str()))
        return false;

    mesh = std::move(loaded_mesh);
    tri_renderer.upload(mesh.positions, mesh.triangle_indices);

    tets.clear();
    tet_renderer.clear();

    const tetrahedralizer::Bounds3 bounds = mesh.bounds();
    state.mesh_bounds = bounds;
    tetrahedralizer::frameCamera(state.camera, state.orbit_center, bounds);
    const float diagonal = bounds.getDimensions().length();
    state.scene_scale = diagonal > 0.0f ? diagonal : 1.0f;
    state.camera.speed = kBaseCameraSpeed * state.scene_scale;

    tetrahedralizer::writeLastPath(path);
    glfwSetWindowTitle(window, ("Tetrahedralizer - " + path).c_str());
    return true;
}

void promptLoadMesh(GLFWwindow* window, AppState& state, tetrahedralizer::TriMesh& mesh,
                    tetrahedralizer::TriMeshRenderer& tri_renderer, tetrahedralizer::Tetrahedralizer& tets,
                    tetrahedralizer::TetMeshRenderer& tet_renderer)
{
    std::string path;
    if (FileDialog::getFileName(path, true) &&
        !loadMesh(window, path, state, mesh, tri_renderer, tets, tet_renderer))
        FileDialog::showError("Load Failed", "Failed to load OBJ mesh.");
}

void loadLastMesh(GLFWwindow* window, AppState& state, tetrahedralizer::TriMesh& mesh,
                  tetrahedralizer::TriMeshRenderer& tri_renderer, tetrahedralizer::Tetrahedralizer& tets,
                  tetrahedralizer::TetMeshRenderer& tet_renderer)
{
    std::string path;
    if (!tetrahedralizer::readLastPath(path) ||
        !loadMesh(window, path, state, mesh, tri_renderer, tets, tet_renderer))
        promptLoadMesh(window, state, mesh, tri_renderer, tets, tet_renderer);
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

    GLFWwindow* window = glfwCreateWindow(3200, 1800, "Tetrahedralizer", nullptr, nullptr);
    if (window == nullptr)
    {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    if (GLFWmonitor* monitor = glfwGetPrimaryMonitor())
    {
        if (const GLFWvidmode* mode = glfwGetVideoMode(monitor))
        {
            int window_width = 0;
            int window_height = 0;
            glfwGetWindowSize(window, &window_width, &window_height);
            glfwSetWindowPos(window, (mode->width - window_width) / 2, (mode->height - window_height) / 2);
        }
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
    tetrahedralizer::TriMeshRenderer tri_renderer;
    tetrahedralizer::Tetrahedralizer tets;
    tetrahedralizer::TetMeshRenderer tet_renderer;
    bool show_mesh = true;
    bool show_tets = true;
    bool wireframe = false;
    bool tet_wireframe = false;
    float tet_scale = 0.85f;

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

        const tetrahedralizer::Vec3 clip = clipLimits(state);
        applyClipPlanes(clip);
        if (show_mesh)
            tri_renderer.render(wireframe);
        disableClipPlanes();

        if (show_tets)
        {
            tet_renderer.setClip(clip);
            tet_renderer.render(tet_wireframe, tet_scale);
        }

        ImGui_ImplGLFW_NewFrame();
        ImGui::SetNextWindowSize(ImVec2(320.0f, 600.0f), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Controls");
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float half_width = (ImGui::GetContentRegionAvail().x - spacing) * 0.5f;
        if (viewer::imgui_widgets::button("Load OBJ", half_width))
            promptLoadMesh(window, state, mesh, tri_renderer, tets, tet_renderer);
        ImGui::SameLine();
        if (viewer::imgui_widgets::button("Load last", half_width))
            loadLastMesh(window, state, mesh, tri_renderer, tets, tet_renderer);

        if (!mesh.empty())
        {
            viewer::imgui_widgets::section_separator();
            viewer::imgui_widgets::section_heading("Mesh");
            viewer::imgui_widgets::checkbox("Show mesh", &show_mesh);
            viewer::imgui_widgets::checkbox("Wireframe", &wireframe);

            viewer::imgui_widgets::section_separator();
            viewer::imgui_widgets::section_heading("Tets");
            viewer::imgui_widgets::slider_float("Voxel size", &state.voxel_spacing, 0.01f, 0.1f);
            viewer::imgui_widgets::slider_int("Hole close", &state.hole_close_radius, 0, 5);
            viewer::imgui_widgets::slider_int("Smooth iters", &state.smoothing_iterations, 0, 50);
            viewer::imgui_widgets::slider_float("Volume factor", &state.volume_factor, 0.1f, 1.5f);
            viewer::imgui_widgets::checkbox("Cut with input mesh", &state.cut_with_input_mesh);
            if (viewer::imgui_widgets::button_full_width("Create tets"))
                createTets(mesh, state.voxel_spacing, state.hole_close_radius, state.smoothing_iterations,
                           state.volume_factor, state.cut_with_input_mesh, tets, tet_renderer);
            viewer::imgui_widgets::checkbox("Show tets", &show_tets);
            viewer::imgui_widgets::checkbox("Tet wireframe", &tet_wireframe);
            if (!tet_wireframe)
                viewer::imgui_widgets::slider_float("Tet scale", &tet_scale, 0.5f, 1.0f);

            viewer::imgui_widgets::section_separator();
            viewer::imgui_widgets::slider_int("Clip X", &state.clip[0], 0, 100);
            viewer::imgui_widgets::slider_int("Clip Y", &state.clip[1], 0, 100);
            viewer::imgui_widgets::slider_int("Clip Z", &state.clip[2], 0, 100);
        }

        ImGui::End();
        ImGui::Render();
        ImGui_ImplGLFW_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    tri_renderer.clear();
    tet_renderer.clear();
    ImGui_ImplGLFW_Shutdown();
    ImGui::DestroyContext(imgui_context);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
