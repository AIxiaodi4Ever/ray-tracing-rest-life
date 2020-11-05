#ifndef CONST_MEDIUM_H
#define CONST_MEDIUM_H

#include <curand_kernel.h>
#include "hittable.h"
#include "material.h"
#include "my_texture.h"

class const_medium : public hittable
{
public:
    __device__ const_medium(hittable *b, float d, curandState *local_rand_state, my_texture *a)
        : boundary(b), neg_inv_density(-1 / d), medium_rands(local_rand_state) { phase_function = new isotropic(a); }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const
    {
        return boundary->bounding_box(t0, t1, output_box);
    }

public:
    hittable *boundary;
    float neg_inv_density;
    curandState *medium_rands;
    material *phase_function;
};

__device__ inline bool const_medium::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
// print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && random_float(medium_rands) < 0.0001;

    hit_record rec1, rec2;

    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        return false;
    if (!boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2))
        return false;
    
    if (debugging)
        printf("\nt0=%f, t1=%f\n", rec1.t, rec2.t);
    
    if (rec1.t < t_min)
        rec1.t = t_min;
    if (rec2.t > t_max)
        rec2.t = t_max;
    
    if (rec1.t >= rec2.t)
        return false;
    
    if (rec1.t < 0)
        rec1.t = 0;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const float hit_distance = neg_inv_density * log(random_float(medium_rands));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    if (debugging)
        printf("hit_distance = %f\nrec.t = %f\nrec.p = %f\n", hit_distance, rec.t, rec.p);

    // 反射方向是随机的，仅由phase_function决定
    rec.normal = vec3(1, 0, 0);     // arbitrary
    rec.front_face = true;          // alse_= arbitrary
    rec.mat_ptr = phase_function;

    return true;
}

#endif