/* this file define 3-dimension vector and it's operational rule */
#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <math.h>       // 使用的是CUDA自带的math库，这个应该没用

class vec3 {
public:
    __host__ __device__ vec3() : e{0,0,0} { }
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} { }
    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(const float t)
    {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

public:
    float e[3];
};

// vec3 Utility Functions
inline std::ostream& operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3 &v, float t)
{
    return (1 / t) * v;
}

__host__ __device__ inline bool operator==(const vec3 &lhs, const vec3 &rhs)
{
    return (lhs.e[0] == rhs.e[0]) && (lhs.e[1] == rhs.e[1]) && (lhs.e[2] == rhs.e[2]);
}

__host__ __device__ inline bool operator!=(const vec3 &lhs, const vec3 &rhs)
{
    return !(lhs == rhs);
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
    return (u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

#endif
