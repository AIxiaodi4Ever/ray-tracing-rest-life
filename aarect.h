#ifndef AARECT_H
#define AARECT_H

#include "hittable.h"

// 平行于xy的矩形
class xy_rect : public hittable
{
public:
    __device__ xy_rect() {}
    __device__ xy_rect(material *mat, float _x0, float _x1, float _y0, float _y1, float _k) :
        hittable(mat), x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k){}

    __device__ virtual bool hit(const ray &r, float t0, float t1, hit_record &rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const
    {
        output_box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
        return true;
    }

public:
    float x0, x1, y0, y1, k;
};

__device__ inline bool xy_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
    float t = (k - r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

// 平行于xz的矩形
class xz_rect : public hittable
{
public:
    __device__ xz_rect() {}
    __device__ xz_rect(material *mat, float _x0, float _x1, float _z0, float _z1, float _k) :
        hittable(mat), x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k) {}

    __device__ virtual bool hit(const ray &r, float t0, float t1, hit_record &rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const
    {
        output_box = aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
        return true;
    }

    __device__ virtual float pdf_value(const vec3& origin, const vec3& v) const {
        hit_record rec;
        if (!this->hit(ray(origin, v), 0.001, MY_INFINITY, rec))
            return 0;
        float area = (x1-x0)*(z1-z0);
        float distance_squared = rec.t * rec.t * v.length_squared();
        float cosine = fabs(dot(v, rec.normal) / v.length());
        return distance_squared / (cosine * area);
    }

    __device__ virtual vec3 random(const vec3& origin, curandState *local_rand_state) const {
        vec3 random_point = vec3(x0 + (x1 - x0) * random_float(local_rand_state), k, z0 + (z1 - z0) * random_float(local_rand_state));
        return random_point - origin;
    }

public:
    float x0, x1, z0, z1, k;
};

__device__ inline bool xz_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
    float t = (k - r.origin().y()) / r.direction().y();
    if (t < t0 || t > t1)
        return false;
    float x = r.origin().x() + t * r.direction().x();
    float z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

class yz_rect : public hittable
{
public:
    __device__ yz_rect() {}
    __device__ yz_rect(material *mat, float _y0, float _y1, float _z0, float _z1, float _k) :
        hittable(mat), y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k) {}

    __device__ virtual bool hit(const ray &r, float t0, float t1, hit_record &rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const
    {
        output_box = aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
        return true;
    }

public:
    float y0, y1, z0, z1, k;
};

__device__ inline bool yz_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
    float t = (k - r.origin().x()) / r.direction().x();
    if (t < t0 || t > t1)
        return false;
    float y = r.origin().y() + t * r.direction().y();
    float z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (z - z0) / (z1 - z0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    vec3 outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

#endif