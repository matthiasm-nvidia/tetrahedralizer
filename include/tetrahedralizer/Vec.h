#pragma once

#include <cmath>

namespace tetrahedralizer
{

constexpr float MaxFloat = 3.402823466e+38f;

enum _ZERO
{
    Zero
};
enum _IDENTITY
{
    Identity
};
enum _EMPTY
{
    Empty
};

inline float Min(float f0, float f1)
{
    return f0 < f1 ? f0 : f1;
}

inline float Max(float f0, float f1)
{
    return f0 > f1 ? f0 : f1;
}

struct Vec3
{
    Vec3() : x(0.0f), y(0.0f), z(0.0f)
    {
    }
    Vec3(_ZERO) : x(0.0f), y(0.0f), z(0.0f)
    {
    }
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_)
    {
    }

    float& operator[](unsigned int index)
    {
        return reinterpret_cast<float*>(this)[index];
    }
    const float& operator[](unsigned int index) const
    {
        return reinterpret_cast<const float*>(this)[index];
    }

    bool operator==(const Vec3& v) const
    {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator!=(const Vec3& v) const
    {
        return !(*this == v);
    }

    bool isZero() const
    {
        return x == 0.0f && y == 0.0f && z == 0.0f;
    }

    void setMin(const Vec3& v)
    {
        x = x < v.x ? x : v.x;
        y = y < v.y ? y : v.y;
        z = z < v.z ? z : v.z;
    }

    void setMax(const Vec3& v)
    {
        x = x > v.x ? x : v.x;
        y = y > v.y ? y : v.y;
        z = z > v.z ? z : v.z;
    }

    Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }
    Vec3 operator+(const Vec3& v) const
    {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    Vec3 operator-(const Vec3& v) const
    {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    void operator+=(const Vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }
    void operator-=(const Vec3& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }
    void operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
    }
    void operator/=(float f)
    {
        x /= f;
        y /= f;
        z /= f;
    }
    Vec3 operator*(float f) const
    {
        return Vec3(x * f, y * f, z * f);
    }
    Vec3 operator/(float f) const
    {
        return Vec3(x / f, y / f, z / f);
    }

    float magnitude() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }
    float magnitudeSquared() const
    {
        return x * x + y * y + z * z;
    }
    float length() const
    {
        return magnitude();
    }
    float normalize()
    {
        const float m = magnitude();
        if (m > 0.0f)
            *this /= m;
        return m;
    }
    float dot(const Vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }
    Vec3 cross(const Vec3& v) const
    {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    float x, y, z;
};

inline Vec3 operator*(float f, const Vec3& v)
{
    return Vec3(f * v.x, f * v.y, f * v.z);
}

struct Vec4
{
    Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
    {
    }
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_)
    {
    }

    float x, y, z, w;
};

struct Ray
{
    Ray() : orig(Zero), dir(Zero)
    {
    }
    Ray(const Vec3& orig_, const Vec3& dir_) : orig(orig_), dir(dir_)
    {
    }

    Vec3 at(float t) const
    {
        return orig + t * dir;
    }

    Vec3 orig;
    Vec3 dir;
};

struct Quat;

struct Mat33
{
    Mat33() : column0(Zero), column1(Zero), column2(Zero)
    {
    }
    Mat33(const Vec3& c0, const Vec3& c1, const Vec3& c2) : column0(c0), column1(c1), column2(c2)
    {
    }
    Mat33(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
        : column0(Vec3(a00, a10, a20)), column1(Vec3(a01, a11, a21)), column2(Vec3(a02, a12, a22))
    {
    }

    Mat33 getTranspose() const
    {
        return Mat33(Vec3(column0.x, column1.x, column2.x), Vec3(column0.y, column1.y, column2.y),
                     Vec3(column0.z, column1.z, column2.z));
    }

    Vec3 operator*(const Vec3& v) const
    {
        return column0 * v.x + column1 * v.y + column2 * v.z;
    }

    Mat33 operator*(const Mat33& M) const
    {
        return Mat33(*this * M.column0, *this * M.column1, *this * M.column2);
    }

    Vec3 column0, column1, column2;
};

struct Quat
{
    Quat() : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
    {
    }
    Quat(float angleRadians, const Vec3& unitAxis)
    {
        const float a = angleRadians * 0.5f;
        const float s = std::sin(a);
        w = std::cos(a);
        x = unitAxis.x * s;
        y = unitAxis.y * s;
        z = unitAxis.z * s;
    }

    void normalize()
    {
        const float len = std::sqrt(x * x + y * y + z * z + w * w);
        if (len > 0.0f)
        {
            const float invLen = 1.0f / len;
            x *= invLen;
            y *= invLen;
            z *= invLen;
            w *= invLen;
        }
    }

    Vec3 rotate(const Vec3& v) const
    {
        const float vx = 2.0f * v.x;
        const float vy = 2.0f * v.y;
        const float vz = 2.0f * v.z;
        const float w2 = w * w - 0.5f;
        const float dot2 = (x * vx + y * vy + z * vz);
        return Vec3((vx * w2 + (y * vz - z * vy) * w + x * dot2), (vy * w2 + (z * vx - x * vz) * w + y * dot2),
                    (vz * w2 + (x * vy - y * vx) * w + z * dot2));
    }

    Quat operator*(const Quat& q) const
    {
        return Quat(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
                    w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);
    }

    // Temporary ctor used by operator* above before assign — keep members public.
    Quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_)
    {
    }

    float x, y, z, w;
};

struct Bounds3
{
    Bounds3() : minimum(Zero), maximum(Zero)
    {
    }
    Bounds3(_EMPTY)
    {
        setEmpty();
    }

    void setEmpty()
    {
        minimum = Vec3(MaxFloat, MaxFloat, MaxFloat);
        maximum = Vec3(-MaxFloat, -MaxFloat, -MaxFloat);
    }

    bool isEmpty() const
    {
        return minimum.x > maximum.x;
    }

    void include(const Vec3& p)
    {
        minimum.setMin(p);
        maximum.setMax(p);
    }

    Vec3 getDimensions() const
    {
        return maximum - minimum;
    }

    Vec3 getHalfExtents() const
    {
        return getDimensions() * 0.5f;
    }

    Vec3 getCenter() const
    {
        return (maximum + minimum) * 0.5f;
    }

    Vec3 minimum, maximum;
};

inline float closestPointOnRay(const Vec3& p, const Ray& ray)
{
    if (ray.dir == Vec3(Zero))
        return 0.0f;
    return ray.dir.dot(p - ray.orig) / ray.dir.magnitudeSquared();
}

} // namespace tetrahedralizer
