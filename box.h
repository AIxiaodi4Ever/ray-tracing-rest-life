#ifndef BOX_H
#define BOX_H

#include "hittable.h"
#include "hittable_list.h"

class box : public hittable
{
public:
    __device__ box() {}
    __device__ box( material *ptr, const vec3 &p0, const vec3 &p1);

    __device__ virtual bool hit(const ray &r, float t0, float t1, hit_record &rec) const;

    __device__ bool bounding_box(float t0, float t1, aabb& output_box) const
    {
        output_box = aabb(box_min, box_max);
        return true;
    }

public:
    vec3 box_min;
    vec3 box_max;
    hittable_list sides;
};

/* 不敢相信这个可以在__global__里调用并成功运行，而且运行的很快，还是不太了解GPU内存相关的知识 */
__device__ inline box::box(material *ptr, const vec3 &p0, const vec3 &p1)
{
    box_min = p0;
    box_max = p1;

    hittable **box_list = new hittable *[6];
    box_list[0] = new xy_rect(ptr, p0.x(), p1.x(), p0.y(), p1.y(), p1.z());
    box_list[1] = new flip_face(new xy_rect(ptr, p0.x(), p1.x(), p0.y(), p1.y(), p0.z()));

    box_list[2] = new xz_rect(ptr, p0.x(), p1.x(), p0.z(), p1.z(), p1.y());
    box_list[3] = new flip_face(new xz_rect(ptr, p0.x(), p1.x(), p0.z(), p1.z(), p0.y()));

    box_list[4] = new yz_rect(ptr, p0.y(), p1.y(), p0.z(), p1.z(), p1.x());
    box_list[5] = new flip_face(new yz_rect(ptr, p0.y(), p1.y(), p0.z(), p1.z(), p0.x()));

    sides = hittable_list(box_list, 6);
}

__device__ inline bool box::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
    return sides.hit(r, t0, t1, rec);
}

#endif