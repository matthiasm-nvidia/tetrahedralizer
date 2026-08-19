#pragma once

#include "tetrahedralizer/Vec.h"

namespace tetrahedralizer
{

class Camera
{
public:
    void init();
    void reset_view();
    void look_at(const Vec3& position, const Vec3& target);
    void set_up_y();
    void set_up_z();

    void handle_mouse_view(int dx, int dy);
    void handle_mouse_orbit(int dx, int dy, const Vec3& center);
    void handle_mouse_translate(int dx, int dy, float scale = 0.01f);
    void handle_key_down(int key);
    void handle_wheel(int rotation);

    void applyProjection(int width, int height, double far_plane = 10000.0);
    void applyView();

    Vec3 pos;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float speed = 0.1f;
    float fov = 40.0f;

    float proj_mat[16] = {};
    float view_mat[16] = {};
};

void frameCamera(Camera& camera, Vec3& orbit_center, const Bounds3& bounds);

} // namespace tetrahedralizer
