/* 
 * this file define a hittable object moving_sphere and a function inherit from class hittale.h
 * to check if the ray hit the specific moving_sphere
 */
#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "hittable.h"

class moving_sphere: public hittable 
{
public:
    __device__ moving_sphere() {};
    __device__ moving_sphere(material *m, vec3 cen0, vec3 cen1, float t0, float t1, float r) : 
                hittable(m), center0(cen0), center1(cen1),time0(t0), time1(t1), radius(r) {};

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const;
    __device__ vec3 center(float time) const;

public:
    vec3 center0, center1;
    float time0, time1;
    float radius;
};

__device__ vec3 moving_sphere::center(float time) const
{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    // 物体当前的圆心是根据光线发出的时间确定的
    vec3 oc = r.origin() - center(r.time());
    float a = r.direction().length_squared();
    float half_b = dot(r.direction(), oc);
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant > 0)
    {
        float root = sqrt(discriminant);
        float temp = (-half_b - root) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center(r.time())) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center(r.time())) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool moving_sphere::bounding_box(float t0, float t1, aabb &output_box) const
{
    aabb box0(center(t0) - vec3(radius, radius, radius),
              center(t0) + vec3(radius, radius, radius));
    aabb box1(center(t1) - vec3(radius, radius, radius),
              center(t1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

#endif