/* 
 * this file define the ray(eye-ray...) incldue emit-point(orig) and direction(dir) 
 * and a function : vec3 at(double t) const; to get it's state at time t 
 *
 * 
 */
#ifndef RAY_H
#define RAY_H

#include "vec3.h"
#include <iostream>

using namespace std;

class ray {
public:
    __device__ ray() {}
    __device__ ray(const vec3 &origin, const vec3 &direction, float time = 0.0) : 
                orig(origin), dir(direction), tm(time) { }

    __device__ vec3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }
    __device__ float time() const { return tm; }

    __device__ vec3 at(float t) const 
    {
        return orig + t * dir;
    }

public: 
    vec3 orig;
    vec3 dir;
    float tm;
};

#endif