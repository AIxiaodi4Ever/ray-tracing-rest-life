#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include "RTnextweek.h"
#include "my_texture.h"

// ��������ʽ��Christophe Schlick���ƣ���ò�ͬ������µķ�����
__device__ float schlick(float cosine, float ref_idx)
{
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 * (1 - r0) * pow(1 - cosine, 5);
}

// ��õ�����е���������
__device__ inline bool refract(const vec3& v, const vec3& n, float etai_over_etat, vec3& refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - etai_over_etat*etai_over_etat*(1-dt*dt);
    if (discriminant > 0) {
        refracted = etai_over_etat*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

// ���ؾ���������
// ���۷�����ָ���ڻ��ⶼ�ܵõ���ȷ�ķ��䷽��
__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}


class material {
public:
    __device__ virtual ~material(){}

    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const
    {
        return vec3(0, 0, 0);
    }

    // ������Ϊ���ߴ������ٶ�Ϊ��������ԽӴ�ʱ���ߵ�ʱ���뷴����ʱ�䶼���ڹ�����������ʱ��
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, 
                                    float &pdf, curandState *local_rand_state) const
    {
        return false;
    }

    // ���˼�virtual������һ������
    __device__ virtual float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
    {
        return 0;
    }
};

// ���������طֲ��ı���
class lambertian : public material {
public:
    __device__ lambertian(my_texture *a) : albedo(a) {}

    __device__ ~lambertian() { delete albedo; }

    __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, float& pdf, curandState *local_rand_state) 
    const
    {
        vec3 target = rec.p + rec.normal + random_unit_vector(local_rand_state);
        scattered = ray(rec.p, unit_vector(target - rec.p), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        pdf = dot(rec.normal, unit_vector(scattered.direction())) / M_PI;
        return true;
    }

    __device__ virtual float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
    {
        float cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return (cosine < 0 ? 0 : cosine / M_PI);
        //return cosine / M_PI;
    }

public:
    //vec3 albedo;
    my_texture *albedo;
};

// ��������
class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const
    {
        // dir���ܲ��ǵ�λ���������Ե���unit_vector()
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        // ������scattered������scatter......
        // ray scatter = ray(rec.p, reflected);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

public:
    vec3 albedo;        // �����ʣ�����RGB�����ֱ𱻷������
    float fuzz;        // ͨ��΢С�ı�������䷽�����ı����ӽ����ֲڶ�
};

// ����ʣ�������ˮ�����ʯ�ȣ�
class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ virtual bool scatter(const ray&r_in, const hit_record& rec, vec3& attenuation, ray& scattered, 
        curandState *local_rand_state) const
    {   // �������ͬһ��ķ�����
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) 
        {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else 
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());
        return true;
    }

public:
    float ref_idx;
};

// ���ȹ�Դ
class diffuse_light : public material
{
public:
    __device__ diffuse_light(my_texture *a) : emit(a) {}

    __device__ ~diffuse_light() { delete emit; }

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered,
        curandState *local_rand_state) const
    {
        return false;
    }

    __device__ virtual vec3 emitted(float u, float v, const vec3& p) const 
    {
        return emit->value(u, v, p);
    }

public:
    my_texture *emit;
};


// ����ͬ�Խ���
class isotropic : public material
{
public:
    __device__ isotropic(my_texture *a) : albedo(a) {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record& rec, vec3 &attenuation, ray& scattered,
        curandState *local_rand_state) const
    {
        scattered = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

public:
    my_texture *albedo;
};

#endif