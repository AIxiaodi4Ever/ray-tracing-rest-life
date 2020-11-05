/* this file define camera class from camera we can get ray */
#ifndef CAMERA_H
#define CAMERA_H

#include <curand_kernel.h>
#include "ray.h"
#include "RTnextweek.h"

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, 
                    float aperture, float focus_dist, float t0 = 0.0, float t1 = 0.0) 
    { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        time0 = t0;
        time1 = t1;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) 
    {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, 
                   lower_left_corner + s*horizontal + t*vertical - origin - offset,
                   (time1 - time0) * curand_uniform(local_rand_state) + time0);
    }

public:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
    float time0, time1;         // shutter open/close times
};

#endif