#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include "RTnextweek.h"
#include "my_texture.h"
#include "onb.h"
#include "pdf.h"

// 菲涅尔公式的Christophe Schlick近似，获得不同入射角下的反射率
__device__ float schlick(float cosine, float ref_idx)
{
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 * (1 - r0) * pow(1 - cosine, 5);
}

// 获得电解质中的折射向量
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

// 返回镜反射向量
// 无论法向量指向内或外都能得到正确的反射方向
__device__ inline vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

struct scatter_record {
    ray specular_ray;
    bool is_specular;
    vec3 attenuation;
    pdf *pdf_ptr;
};

class material {
public:
    __device__ virtual ~material(){}

    __device__ virtual vec3 emitted(const ray& r_in, const hit_record& rec, float u, float v, const vec3 &p) const
    {
        return vec3(0, 0, 0);
    }

    // 这里认为光线传播的速度为无穷大所以接触时光线的时间与反射后的时间都等于光线最初发射的时间
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) const
    {
        return false;
    }

    // 忘了加virtual导致了一个错误
    __device__ virtual float scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
    {
        return 0;
    }
};

// 服从兰贝特分布的表面
class lambertian : public material {
public:
    __device__ lambertian(my_texture *a) : albedo(a) {}

    __device__ ~lambertian() { delete albedo; }

    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) 
    const
    {
        srec.is_specular = false;
        srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = new cosine_pdf(rec.normal);
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

// 金属表面
class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) const
    {
        // dir可能不是单位向量，所以调用unit_vector()
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        // 这里是scattered而不是scatter......
        // ray scatter = ray(rec.p, reflected);
        srec.specular_ray = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        srec.attenuation = albedo;
        srec.is_specular = true;
        srec.pdf_ptr = 0;
        return true;
    }

public:
    vec3 albedo;        // 反射率，决定RGB分量分别被反射多少
    float fuzz;        // 通过微小改变金属反射方向来改变增加金属粗糙度
};

// 电解质（玻璃、水、金刚石等）
/* 透明介质如球，可以在电解质球内部添加一个半径为负的电解质球实现， 
*  所以同理其他形状的透明介质只需添加一个尺寸略小的电解质，但每个面的法向量都与原来相反即可*/
class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) const
    {   // 与光线在同一侧的法向量
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        srec.attenuation = vec3(1.0, 1.0, 1.0);
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
            srec.specular_ray = ray(rec.p, reflected, r_in.time());
        else
            srec.specular_ray = ray(rec.p, refracted, r_in.time());
        srec.is_specular = true;
        return true;
    }

public:
    float ref_idx;
};

// 均匀光源
class diffuse_light : public material
{
public:
    __device__ diffuse_light(my_texture *a) : emit(a) {}

    __device__ ~diffuse_light() { delete emit; }

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) const
    {
        return false;
    }

    __device__ virtual vec3 emitted(const ray& r_in, const hit_record& rec, float u, float v, const vec3& p) const 
    {
        // 不知道传入的r_in有什么用
        // flip_face终于派上用场了
        if (rec.front_face)
            return emit->value(u, v, p);
        else
            return vec3(0, 0, 0);
    }

public:
    my_texture *emit;
};


// 各向同性介质
class isotropic : public material
{
public:
    __device__ isotropic(my_texture *a) : albedo(a) {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, scatter_record& srec, curandState *local_rand_state) const
    {
        /*scattered = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);*/
        return true;
    }

public:
    my_texture *albedo;
};

#endif