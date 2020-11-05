/* 
 * this file define a hittable object sphere and a function inherit from class hittale.h
 * to check if the ray hit the specific sphere
 */
#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

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

public:
    vec3 center;
    float radius;
};

// ����������������춥�Ǽ����Բ�ܽ�
__device__ inline void get_sphere_uv(const vec3 &p, float &u, float &v)
{
    // atan2����ֵ��Χ[-pi, pi];
    float phi = atan2(p.z(), p.x());
    // asin���ط�Χ[-pi/2, pi /2];
    float theta = asin(p.y());  
    u = 1 - (phi + M_PI) / (2 * M_PI);          // x����z����x����Ϊ0~0.5��x����z�����ٵ�x����Ϊ0.5~1
    v = (theta + M_PI / 2) / M_PI;              // y����y����Ϊ0~1.0
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

#endif