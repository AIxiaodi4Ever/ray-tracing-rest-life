/* 
 * this file define a hittable object sphere and a function inherit from class hittale.h
 * to check if the ray hit the specific sphere
 */
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "onb.h"
#include "pdf.h"

class sphere : public hittable {
public:
    __device__ sphere() {};
    __device__ sphere(material *m, vec3 cen, float r) : hittable(m), center(cen), radius(r) {};

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const
    {
        output_box = aabb(center - vec3(radius, radius, radius),
                          center + vec3(radius, radius, radius));
        return true;
    }

    __device__ virtual float pdf_value(const vec3 &o, const vec3 &v) const;

    __device__ virtual vec3 random(const vec3 &o, curandState *local_rand_state) const;

public:
    vec3 center;
    float radius;
};

// 返回球坐标下相对天顶角及相对圆周角
__device__ inline void get_sphere_uv(const vec3 &p, float &u, float &v)
{
    // atan2返回值范围[-pi, pi];
    float phi = atan2(p.z(), p.x());
    // asin返回范围[-pi/2, pi /2];
    float theta = asin(p.y());  
    u = 1 - (phi + M_PI) / (2 * M_PI);          // x负向到z负向到x正向为0~0.5，x正向到z正向再到x负向为0.5~1
    v = (theta + M_PI / 2) / M_PI;              // y负向到y正向为0~1.0
}

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0)
    {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
            return true;
        }
    }
    return false;
}

__device__ float sphere::pdf_value(const vec3 &o, const vec3 &v) const
{
    hit_record rec;
    if (!this->hit(ray(o, v), 0.001, MY_INFINITY, rec))
        return 0;
    float cos_theta_max = sqrt(1 - radius * radius / (center - o).length_squared());
    float solid_angle = 2 * M_PI * (1 - cos_theta_max);
    return 1 / solid_angle;
}

__device__ vec3 sphere::random(const vec3 &o, curandState *local_rand_state) const
{
    vec3 direction = center - o;
    float distance_squared = direction.length_squared();
    onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(radius, distance_squared, local_rand_state));
}

#endif