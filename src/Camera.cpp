#include "tetrahedralizer/Camera.h"

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <cmath>

namespace tetrahedralizer
{
namespace
{
constexpr float k_degrees_per_pixel = 0.5f;
constexpr float k_pi = 3.14159265358979323846f;

void set_identity(float (&matrix)[16])
{
    for (int i = 0; i < 16; ++i)
        matrix[i] = i % 5 == 0 ? 1.0f : 0.0f;
}

int positive_or_one(int value)
{
    return value <= 0 ? 1 : value;
}
} // namespace

void Camera::init()
{
    up = Vec3(0.0f, 1.0f, 0.0f);
    speed = 0.1f;
    fov = 40.0f;
    reset_view();
    set_identity(proj_mat);
    set_identity(view_mat);
}

void Camera::reset_view()
{
    if (up.z == 0.0f)
    {
        pos = Vec3(0.0f, 2.0f, 10.0f);
        forward = Vec3(0.0f, 0.0f, -1.0f);
    }
    else
    {
        pos = Vec3(10.0f, 2.0f, 0.0f);
        forward = Vec3(-1.0f, 0.0f, 0.0f);
    }
    right = forward.cross(up);
    right.normalize();
}

void Camera::look_at(const Vec3& position, const Vec3& target)
{
    pos = position;
    forward = target - pos;
    forward.normalize();
    up = Vec3(0.0f, 1.0f, 0.0f);
    right = forward.cross(up);
    right.normalize();
    up = right.cross(forward);
    up.normalize();
}

void Camera::set_up_y()
{
    up = Vec3(0.0f, 1.0f, 0.0f);
    reset_view();
}

void Camera::set_up_z()
{
    up = Vec3(0.0f, 0.0f, 1.0f);
    reset_view();
}

void Camera::handle_mouse_view(int dx, int dy)
{
    forward.normalize();
    right = forward.cross(up);
    right.normalize();

    Quat qx(k_pi * static_cast<float>(-dx) * k_degrees_per_pixel / 180.0f, up);
    forward = qx.rotate(forward);
    Quat qy(k_pi * static_cast<float>(-dy) * k_degrees_per_pixel / 180.0f, right);
    forward = qy.rotate(forward);
    forward.normalize();
    up = right.cross(forward);
    up.normalize();
}

void Camera::handle_mouse_translate(int dx, int dy, float scale)
{
    pos -= right * scale * static_cast<float>(dx);
    pos += up * scale * static_cast<float>(dy);
}

void Camera::handle_mouse_orbit(int dx, int dy, const Vec3& center)
{
    Mat33 initial_basis;
    initial_basis.column0 = right;
    initial_basis.column1 = forward;
    initial_basis.column2 = up;

    up.normalize();
    right.normalize();

    Quat q(static_cast<float>(-dx) * 0.01f, up);
    q = q * Quat(static_cast<float>(-dy) * 0.01f, right);
    q.normalize();

    forward = q.rotate(forward);
    up = q.rotate(up);
    right = q.rotate(right);

    right.y = 0.0f;
    right.normalize();
    up = right.cross(forward);
    up.normalize();
    forward = up.cross(right);
    forward.normalize();

    Mat33 rotated_basis;
    rotated_basis.column0 = right;
    rotated_basis.column1 = forward;
    rotated_basis.column2 = up;

    const Mat33 delta = rotated_basis * initial_basis.getTranspose();
    pos = center + delta * (pos - center);
}

void Camera::handle_key_down(int key)
{
    if (key == GLFW_KEY_W)
        pos += forward * speed;
    if (key == GLFW_KEY_S)
        pos -= forward * speed;
    if (key == GLFW_KEY_A)
        pos -= right * speed;
    if (key == GLFW_KEY_D)
        pos += right * speed;
    if (key == GLFW_KEY_E)
        pos -= up * speed;
    if (key == GLFW_KEY_Q)
        pos += up * speed;
}

void Camera::handle_wheel(int rotation)
{
    pos += static_cast<float>(rotation) * forward * speed;
}

void Camera::applyProjection(int width, int height, double far_plane)
{
    const double aspect =
        static_cast<double>(positive_or_one(width)) / static_cast<double>(positive_or_one(height));
    const double near_plane = 0.1;
    const double top = near_plane * std::tan(static_cast<double>(fov) * k_pi / 360.0);
    const double right_extent = top * aspect;
    glFrustum(-right_extent, right_extent, -top, top, near_plane, far_plane);
    glGetFloatv(GL_PROJECTION_MATRIX, proj_mat);
}

void Camera::applyView()
{
    Vec3 view_forward = forward;
    view_forward.normalize();

    Vec3 view_right = view_forward.cross(up);
    if (view_right.normalize() == 0.0f)
    {
        view_right = right;
        view_right.normalize();
    }

    Vec3 view_up = view_right.cross(view_forward);
    view_up.normalize();

    forward = view_forward;
    right = view_right;
    up = view_up;

    const float view[] = {
        right.x,   up.x,   -forward.x,  0.0f,
        right.y,   up.y,   -forward.y,  0.0f,
        right.z,   up.z,   -forward.z,  0.0f,
        -right.dot(pos), -up.dot(pos), forward.dot(pos), 1.0f,
    };

    glLoadMatrixf(view);
    glGetFloatv(GL_MODELVIEW_MATRIX, view_mat);
}

void frameCamera(Camera& camera, Vec3& orbit_center, const Bounds3& bounds)
{
    if (bounds.isEmpty())
        return;

    const Vec3 center = bounds.getCenter();
    const Vec3 half_extents = bounds.getHalfExtents();
    const float radius = half_extents.magnitude();
    const float distance = radius > 0.0f ? radius * 1.2f : 3.0f;
    const Vec3 eye = center + Vec3(distance, distance * 0.5f, distance);
    camera.look_at(eye, center);
    orbit_center = center;
}

} // namespace tetrahedralizer
