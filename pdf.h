#ifndef PDF_H
#define PDF_H

#include <curand_kernel.h>
#include "vec3.h"
#include "onb.h"
#include "hittable.h"
#include "RTnextweek.h"

class pdf {
public:
    __device__ virtual ~pdf(){};

    __device__ virtual double value(const vec3 &direction) const = 0;
    __device__ virtual vec3 generate(curandState *local_rand_state) const = 0;
};

class cosine_pdf : public pdf {
public:
    __device__ cosine_pdf(const vec3 &w) { uvw.build_from_w(w); }

    __device__ virtual double value(const vec3 &direction) const
    {
        float cosine = dot(unit_vector(direction), uvw.w());
        return (cosine <= 0) ? 0 : cosine / M_PI;
    }

    __device__ virtual vec3 generate(curandState *local_rand_state) const
    {
        return uvw.local(random_cosine_direction(local_rand_state));
    }

public: 
    onb uvw;
};

class hittable_pdf : public pdf {
public:
    __device__ hittable_pdf(hittable* p, const vec3& origin) : ptr(p), o(origin) {}
    // 如果有析构函数会导致CUDA error
    __device__ virtual double value(const vec3& direction) const {
        return ptr->pdf_value(o, direction);
    }
    __device__ virtual vec3 generate(curandState *local_rand_state) const {
        return ptr->random(o, local_rand_state);
    }
public:
    vec3 o;
    hittable* ptr;
};

#endif