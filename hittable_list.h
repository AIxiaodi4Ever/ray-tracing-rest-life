/*
 * this file define a list that contain all the hittable object.
 * Define function hit() to iteratively call the hit() function of those
 * hittable object to get the closest hitted object and record the hit point in hit_record
*/

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
public:
    __device__ hittable_list() { };
    __device__ hittable_list(hittable **l, int n) {list = l; list_size = n;}

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const;

    __device__ virtual float pdf_value(const vec3& o, const vec3& v) const;

    __device__ virtual vec3 random(const vec3& o, curandState *local_rand_state) const;

public:
    hittable **list;
    int list_size;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; ++i)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

// 返回列表内所有物体边界的并集 
// ***还不知道应不应该是__device__***
__device__ bool hittable_list::bounding_box(float t0, float t1, aabb &output_box) const
{
    int nums = sizeof(list) / sizeof(hittable *);
    if (sizeof(list) == 0)
        return false;

    aabb temp_box;
    bool first_box = true;

    for (int i = 0; i < nums; ++i)
    {
        // 如果某一个物体没有边界，直接false？
        if (!list[i]->bounding_box(t0, t1, temp_box))
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }
    return true;
}

__device__ float hittable_list::pdf_value(const vec3& o, const vec3& v) const
{
    float weight = 1.0 / list_size;
    float sum = 0.0;

    for (int i = 0; i < list_size; ++i)
    {
        sum += weight * list[i]->pdf_value(o, v);
    }
    return sum;
}

__device__ vec3 hittable_list::random(const vec3& o, curandState *local_rand_state) const
{
    int random_index = (int)(random_float(local_rand_state) * (list_size - 1));
    return list[random_index]->random(o, local_rand_state);
}

#endif