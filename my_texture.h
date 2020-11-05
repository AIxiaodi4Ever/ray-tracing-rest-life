#ifndef MY_TEXTURE_H
#define MY_TEXTURE_H

#include "RTnextweek.h"
#include "perlin.h"

class my_texture
{
public:
    __device__ virtual ~my_texture() { }

    __device__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};

// 纯色纹理
class const_texture : public my_texture
{
public:
    __device__ const_texture() {}
    __device__ const_texture(vec3 c) : color(c) {  }

    __device__ virtual vec3 value(float u, float v, const vec3 &p) const
    {
        return color;
    }

public:
    vec3 color;
};

// 棋盘纹理
class checker_texture : public my_texture
{
public:
    __device__ checker_texture() {}
    __device__ checker_texture(my_texture *t0, my_texture *t1) : even(t0), odd(t1) {}
    __device__ ~checker_texture() { 
        delete even;
        delete odd;
    }

    __device__ virtual vec3 value(float u, float v, const vec3 &p) const
    {
        // 虽然传递了u,v但并没有用到
        float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }
public:
    my_texture *even;
    my_texture *odd;
};

// noise_texture
class noise_texture : public my_texture
{
public:
    __device__ noise_texture(curandState *local_rand_state, float sc) : noise(perlin(local_rand_state)), scale(sc) {}

    __device__ virtual vec3 value(float u, float v, const vec3 &p) const
    {
        //return vec3(1, 1, 1) * 0.5 * (1.0 + noise.noise(scale * p));
        //return vec3(1, 1, 1) * noise.turb(p);
        return vec3(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
    }
public:
    perlin noise;
    float scale;
};

// 图像纹理
class image_texture : public my_texture
{
public:
    __device__ image_texture() {}
    __device__ image_texture(unsigned char *pixels, int A, int B)
        : data(pixels), nx(A), ny(B) {}
    __device__ ~image_texture() 
    {
        delete data;
    }       

    __device__ virtual vec3 value(float u, float v, const vec3& p) const 
    {
        // If we have no texture data, then always emit cyan (as a debugging aid).
        if (data == nullptr)
            return vec3(0,1,1);

        /*
        i表示图像的列，j表示图像的行
        */
        auto i = (int)((u)*nx);
        // 从下到上存储，所以1-v
        auto j = (int)((1-v)*ny-0.001);

        // 按照映射关系图像的中心被映射到从x正向向负向看所对应的位置
        // -x~-z~x~z~-x为0-0.25-0.5-0.75-1.0
        // -y~y为0-1
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i > nx-1) i = nx-1;
        if (j > ny-1) j = ny-1;

        auto r = (int)(data[3*i + 3*nx*j + 0]) / 255.0;
        auto g = (int)(data[3*i + 3*nx*j + 1]) / 255.0;
        auto b = (int)(data[3*i + 3*nx*j + 2]) / 255.0;

        return vec3(r, g, b);
    }
public:
    unsigned char *data;
    int nx, ny;
};

#endif