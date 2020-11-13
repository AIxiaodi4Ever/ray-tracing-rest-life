/*
 * this file define the father class of any hittable object 
 * and define a real virtual function hit() to provide a common interface
 */

#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "aabb.h"

// û����������������޷�����
class material;

struct hit_record
{
    // �����t�ǲ�������ǰ����t���ķ�������ray::dir����ĳ������ײ
    float t;
    float u;    // u,v�洢����λ��ռ���������İٷ���
    float v;
    vec3 p;
    bool front_face;
    vec3 normal;
    material *mat_ptr;


    __device__ void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
public:
    __device__ hittable() { };
    __device__ hittable(material *m) : mat_ptr(m) {  }
    __device__ virtual ~hittable() { delete mat_ptr; }
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;     
    // ����ĳ����ײ����ı߽磨ƽ�������壩
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const = 0;

public:
    // material��ָ���ƶ���hittable�����global����free_word�޷���ȷ���ָ��
    material *mat_ptr;
};

// ��front_faceȡ��������ԭ�����õ�����x��y��z��������������ڱ���Ϊ����x��y��z�ĸ�����
// ��diffuse_light���У��������ù�Դֻ��front_faceΪtrue���淢��
class flip_face : public hittable
{
public:
    __device__ flip_face(hittable *p) : ptr(p) {}

    __device__ ~flip_face() { delete ptr; }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const
    {
        if (!ptr->hit(r, t_min, t_max, rec))
            return false;

        rec.front_face = !rec.front_face;
        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& outpt_box) const
    {
        return ptr->bounding_box(t0, t1, outpt_box);
    }

public:
    hittable *ptr;
};

// ͨ�������ƶ�����ƽ�����壬������ֱ��ƽ������
class translate : public hittable
{
public:
    __device__ translate(hittable *p, const vec3& displacement) : ptr(p), offset(displacement) { }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const;

public:
    hittable *ptr;
    vec3 offset;
};

__device__ bool translate::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec))
        return false;
    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);
    return true;
}

__device__ bool translate::bounding_box(float t0, float t1, aabb &output_box) const
{
    if (!ptr->bounding_box(t0, t1, output_box))
        return false;

    output_box = aabb(output_box.min() + offset, output_box.max() + offset);
    return true;
}

// ������ת����������y����ת����
class rotate_y : public hittable
{
public:
    __device__ rotate_y(hittable *p, float angle);

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &output_box) const
    {
        output_box = bbox;
        return hasbox;
    }

public:
    hittable *ptr;
    float sin_theta;
    float cos_theta;
    bool hasbox;
    aabb bbox;
};

__device__ rotate_y::rotate_y(hittable *p, float angle) : ptr(p)
{
    float radians = degree_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    vec3 min(MY_INFINITY, MY_INFINITY, MY_INFINITY);
    vec3 max(-MY_INFINITY, -MY_INFINITY, -MY_INFINITY);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
                float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
                float z = j * bbox.max().z() + (1 - j) * bbox.max().z();

                float newx = cos_theta * x + sin_theta * z;
                float newz = -sin_theta * x + cos_theta * z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 2; ++c)
                {
                    min[c] = ffmin(min[c], tester[c]);
                    max[c] = ffmax(min[c], tester[c]);
                }//c
            }//k
        }//j
    }//i
    bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray&r, float t_min, float t_max, hit_record& rec) const
{
    vec3 origin = r.origin();
    vec3 direction = r.direction();

    // ������ת����
    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];
    // ������ת���߷���
    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

    ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    vec3 p = rec.p;
    vec3 normal = rec.normal;

    // ������תײ�����ײ�����ⷨ��
    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];
    normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}   

#endif