#ifndef RTNEXTWEEK_H
#define RTNEXTWEEK_H

#include <curand_kernel.h>
#include "vec3.h"

#define M_PI 3.1415926535897932385
#define MY_INFINITY FLT_MAX

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ float degree_to_radians(float angle)
{
    return M_PI * angle / 180;
}

__device__ float random_float(curandState *local_rand_state)
{
    return curand_uniform(local_rand_state);
}

// 单位球内随机取(反射)向量（伪兰贝特分布）
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

// 获得单位圆（光圈）内随机的点
__device__ vec3 random_in_unit_disk(curandState *local_rand_state)
{
    vec3 p;
    while (true)
    {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
        if (p.length_squared() >= 1.0f)
            continue;
        return p;
    }
}

// 获得单位圆表面的随机反射向量（真正的兰贝特分布）
__device__ vec3 random_unit_vector(curandState *local_rand_state) {
    auto a = curand_uniform(local_rand_state) * 2 * M_PI;
    auto z = curand_uniform(local_rand_state) * 2 - 1;
    auto r = sqrt(1 - z * z);
    return vec3(r * cos(a), r * sin(a), z);
}

// 获得半圆内均匀分布的随机反射向量 
__device__ vec3 random_in_hemisphere(const vec3& normal, curand *local_rand_state)
{   // 使用random_unit_vector()得到错误图像，原因未知
    //vec3 in_unit_sphere = random_unit_vector();
    vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__host__ __device__ inline float ffmin(float a, float b) { return a <= b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a >= b ? a : b; }

#endif