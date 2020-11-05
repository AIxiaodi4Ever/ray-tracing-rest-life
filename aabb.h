#ifndef AABB_H
#define AABB_H

#include "RTnextweek.h"
#include "ray.h"

// axis-aligned bounding box
class aabb
{
public:
    __device__ aabb() {}
    __device__ aabb(const vec3 &a, const vec3 &b) 
    { 
        _min = a;
        _max = b;
    }

    __device__ vec3 min() const { return _min; }
    __device__ vec3 max() const { return _max; }

    // 判断是否与有界立方体碰撞，并不是继承自hittable
    __device__ bool hit(const ray &r, float tmin, float tmax) const
    {
        for (int a = 0; a < 3; ++a)
        {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min()[a] - r.origin()[a]) * invD;
            float t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f)
            {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

public:
    // 平行六面体边界
    vec3 _min;
    vec3 _max;
};

// 返回两个边界盒子的并集
__device__ inline aabb surrounding_box(aabb box0, aabb box1)
{
    vec3 small(ffmin(box0.min().x(), box1.min().x()),
               ffmin(box0.min().y(), box1.min().y()),
               ffmin(box0.min().z(), box1.min().z()));

    vec3 big(ffmax(box0.max().x(), box1.max().x()),
               ffmax(box0.max().y(), box1.max().y()),
               ffmax(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}

#endif