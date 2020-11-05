#ifndef BVH_H
#define BVH_H

#include <curand_kernel.h>
#include "hittable.h"
#include "hittable_list.h"

typedef int (*fun_ptr)(const void *a, const void *b);

// 快排的选择枢纽元并分割部分
__device__ int partition(hittable **list, int low, int high, fun_ptr comparator)
{
    hittable* key;
    key = list[low];
    while (low < high)
    {
        while (low < high && comparator(key, list[high]))
            --high;
        if (low < high)
        {
            list[low] = list[high];
            ++low;
        }
        while (low < high && comparator(list[low], key))
            ++low;
        if (low < high)
        {
            list[high] = list[low];
            --high;
        }
    }
    list[low] = key;
    return low;
}

// 用于构造函数的快排程序
__device__ void my_qsort(hittable **list, int start, int end, fun_ptr comparator)
{
    /*int pos;
    if (start < end)
    {
        pos = partition(list, start, end, comparator);
        my_qsort(list, start, pos, comparator);
        my_qsort(list, pos + 1, end, comparator);
    }*/
    int j, p;
    hittable *tmp;
    int n = end - start + 1;
    for (p = 1; p < n; ++p)
    {
        tmp = list[p];
        for (j = p; j > 0 && comparator(tmp, list[j - 1]); --j)
            list[j] = list[j - 1];
        list[j] = tmp;
    }
    return;
}

// bvh_node包含一个边界盒子树，近似满足查找树性质，左边的值小于小于右边，值是边界与坐标轴（任一随机坐标轴）的距离
class bvh_node : public hittable
{
public:
    __device__ bvh_node(){};
    // 虽然传递的是引用，但委托构造函数里使用了指针，因此传递临时量没有影响
    __device__ bvh_node(hittable_list *objs, float time0, float time1, curandState *local_rand_state)
        : bvh_node(objs->list, 0, objs->list_size, time0, time1, local_rand_state) {}
    __device__ bvh_node(hittable **list, size_t start, size_t end, float time0, float time1, curandState *local_rand_state);

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const;

public:
    hittable *left;
    hittable *right;
    aabb box;
};

// 用于qsort的比较函数
__device__ inline bool box_compare(const hittable* a, const hittable* b, int axis)
{
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        printf("no bounding box in bvh_node constructor.\n");

    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ inline int box_x_compare(const void* a, const void* b)
{
    return box_compare((hittable *)a, (hittable *)b, 0);
}

__device__ inline int box_y_compare(const void* a, const void* b)
{
    return box_compare((hittable *)a, (hittable *)b, 1);
}

__device__ inline int box_z_compare(const void* a, const void* b)
{
    return box_compare((hittable *)a, (hittable *)b, 2);
}

__device__ inline bool bvh_node::bounding_box(float t0, float t1, aabb &output_box) const
{
    output_box = box;
    return true;
}

__device__ inline bool bvh_node::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ inline bvh_node::bvh_node(hittable **list, size_t start, size_t end, float time0, float time1, curandState *local_rand_state)
{
    int axis = (int)(curand_uniform(local_rand_state) * 3);

    fun_ptr comparator = (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1)
        left = right = list[start];
    else if (object_span == 2)
    {
        /*if (comparator(list[start], list[start + 1]))
        {
            left = list[start];
            right = list[start + 1];
        }
        else
        {
            left = list[start + 1];
            right = list[start];
        }*/
        my_qsort(list, start, end - 1, comparator);
        left = list[start];
        right = list[start + 1];
    }
    else
    {
       //qsort(list, object_span, sizeof(hittable *), comparator);
       my_qsort(list, start, end - 1, comparator);

       int mid = start + object_span / 2;
       left = new bvh_node(list, start, mid, time0, time1, local_rand_state);
       right = new bvh_node(list, mid, end, time0, time1, local_rand_state);
    }

    aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left)
        || !right->bounding_box(time0, time1, box_right))
    {
        printf("No bounding box in bvh_node construstor.\n");
    }

    box = surrounding_box(box_left, box_right);
}

#endif