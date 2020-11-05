#ifndef PERLIN_H
#define PERLIN_H

#include "RTnextweek.h"

// 三维函数的线性插值
__device__ inline float perlin_interp(vec3 c[2][2][2], float u, float v, float w)
{
    float accum = 0.0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                vec3 weight_v(u - i, v - j, w - k);
                accum +=  (i * u + (1 - i) * (1 - u)) *
                          (j * v + (1 - j) * (1 - v)) *
                          (k * w + (1 - k) * (1 - w)) *
                         dot(c[i][j][k], weight_v);
            }
        }
    }
    return accum;
}

class perlin
{
public:
    __device__ perlin(curandState *local_rand_state) : perlin_rand_state(local_rand_state)
    {
        ranvec = new vec3[point_count];
        for (int i = 0; i < point_count; ++i)
            ranvec[i] = 2 * RANDVEC3 - vec3(1, 1, 1);

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    __device__ ~perlin()
    {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    __device__ float noise(const vec3 &p) const
    {
        float u = p.x() - floor(p.x());
        float v = p.y() - floor(p.y());
        float w = p.z() - floor(p.z());
        // Hermite立方，用于优化线性插值
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);

        // 和255按位与，消除前导0，乘以几，图像的重复周期就减小几倍（小数点后乘以后的进位与舍弃导致）
        int i = floor(p.x());
        int j = floor(p.y());
        int k = floor(p.z());
        vec3 c[2][2][2];

        for (int di = 0; di < 2; ++di)
        {
            for (int dj = 0; dj < 2; ++dj)
            {
                for (int dk = 0; dk < 2; ++dk)
                {
                    c[di][dj][dk] = ranvec[perm_x[i + di & 255] ^
                                             perm_y[j + dj & 255] ^
                                             perm_z[k + dk & 255]];
                }
            }
        }

        return perlin_interp(c, u, v, w);
    }

    __device__ float turb(const vec3 &p, int depth = 7) const 
    {
        float accum = 0.0;
        vec3 temp_p = p;
        float weight = 1.0;

        for (int i = 0; i < depth; ++i)
        {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }

        return fabs(accum);
    }

private:
    static const int point_count = 256;
    curandState *perlin_rand_state;
    vec3 *ranvec;
    int *perm_x;
    int *perm_y;
    int *perm_z;

    // 将perlin_generate_perm和permute声明为static时，无法利用curandState*生成随机数，因为必须得把curandState声明为static，
    // 但声明为static导致其成为 host member
    __device__ int* perlin_generate_perm() 
    {
        int *p = new int[point_count];

        for (int i = 0; i < point_count; ++i)
            p[i] = i;

        permute(p, point_count);

        return p;
    }

    // 必须将其声明为static，否则error: a nonstatic member reference must be relative to a specific object
    __device__ void permute(int *p, int n)
    {
        // 0不需要交换
        for (int i = n - 1; i > 0; --i)
        {
            int target = random_float(perlin_rand_state) * i;
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};

#endif