#include <iostream>
#include <time.h>
#include <float.h>
#define STB_IMAGE_IMPLEMENTATION        // 定义STB_IMAGE_IMPLEMENTATION让头文件只包含函数定义源码，等于将头文件变为.cpp文件
                                        // 否则出现"无法解析的外部符号 stbi_load，函数 main 中引用了该符号"
#include "./stb-master/stb_image.h"
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "aarect.h"
#include "box.h"
#include "moving_sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "pdf.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char *const func, const char *const file, const int line)
{
    if (result)
    {
        cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << "'" << func << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 ray_color(const ray& r, const vec3& background, hittable **d_world, hittable **shape_list, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 cur_emitted = vec3(0, 0, 0);
    scatter_record srec;
    for(int i = 0; i < 50; i++) 
    {
        hit_record rec;
        if ((*d_world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
        {
            float pdf_val;
            vec3 emitted = rec.mat_ptr->emitted(cur_ray, rec, rec.u, rec.v, rec.p);
            if(rec.mat_ptr->scatter(cur_ray, rec, srec, local_rand_state)) 
            {
                cur_emitted = cur_emitted + cur_attenuation * emitted;
                if (srec.is_specular)
                {
                    cur_attenuation =  cur_attenuation * srec.attenuation;
                    cur_ray = srec.specular_ray;
                }
                else
                {
                    // 修改pdf和scattered
                    //hittable_pdf p0(*light_shape, rec.p);
                    hittable_pdf p0(*shape_list, rec.p);
                    mixture_pdf p(&p0, srec.pdf_ptr);
                    // 这里确定反射方向，是否反射向内部是通过rec.mat_ptr->scattering_pdf进行检测的，如果反射向内部，则返回0
                    ray scattered = ray(rec.p, p.generate(local_rand_state), cur_ray.time());
                    pdf_val = p.value(scattered.direction());
                    // this delete can't ignore, otherwise you will encount CUDA error = 700
                    delete srec.pdf_ptr;
                    cur_attenuation = cur_attenuation * srec.attenuation * rec.mat_ptr->scattering_pdf(cur_ray, rec, scattered) / pdf_val;
                    cur_ray = scattered;
                }
            }
            else    
            {
                if (i == 0)
                    return emitted;
                return (cur_emitted + cur_attenuation * emitted); 
            }
        }
        else 
        {
            if (i == 0)
                return background;
            return cur_emitted + cur_attenuation * background;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(2020, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i > max_x || j > max_y) return;
    int pixel_index = j * max_x + i;
    // Each thread get same seed, a different sequence number, no offset
    curand_init(2020, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                hittable **d_world, hittable **shape_list, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = (j * max_x + i);
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    vec3 background(0, 0, 0);
    for (int s = 0; s < ns; ++s)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        vec3 temp = ray_color(r, background, d_world, shape_list, &local_rand_state);
        /* 消除NaN */
        // 必须在这里消除，否则累加后再赋值0.0只能得到黑色的噪点
        // 但这里只能解决NaN导致的黑色噪点，解决不了白色噪点
        if (!(temp[0] == temp[0])) temp[0] = 0.0;
        if (!(temp[1] == temp[1])) temp[1] = 0.0;
        if (!(temp[2] == temp[2])) temp[2] = 0.0;
        // 下面解决无穷大导致的白色噪点，100是随便设的，应该只要大于光源的值就行
        if (temp[0] > 100)  temp[0] = 0.0;
        if (temp[1] > 100)  temp[1] = 0.0;
        if (temp[2] > 100)  temp[2] = 0.0;
        col += temp;
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);      // gamma corrected
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
    /*printf("block index : %d, %d\n", blockIdx.x, blockIdx.y);
    printf("thread index : %d, %d\n", threadIdx.x, threadIdx.y);*/
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, 
            int nx, int ny, curandState *rand_state, unsigned char* d_texture_data, int* d_inxyn) 
{
    // d_world是一个hittable_list（继承自hittable）存储了所有的可碰撞物体，通过调用d_world的hit可以实现对所有物体的遍历以找到最近物体，
    // 提供了一层抽象，否则就需要在ray_color里手动展开所有遍历
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        curandState local_rand_state = *rand_state;

        material* red = new lambertian(new const_texture(vec3(0.65, 0.05, 0.05)));
        material* white = new lambertian(new const_texture(vec3(0.73, 0.73, 0.73)));
        material* green = new lambertian(new const_texture(vec3(0.12, 0.45, 0.15)));

        material* light = new diffuse_light(new const_texture(vec3(15, 15, 15)));   // (15,15,15)

        material* ima = new lambertian(new image_texture(d_texture_data, d_inxyn[0], d_inxyn[1]));
        material* noise = new lambertian(new noise_texture(&local_rand_state, 0.2));

        material* aluminum = new metal(vec3(0.8, 0.85, 0.88), 0.0);

        //material* glass = new dielectric(3);

        d_list[0] = new flip_face(new yz_rect(green, 0, 555, 0, 555, 555)); // green
        d_list[1] = new yz_rect(red, 0, 555, 0, 555, 0);    // red
        // flip_face的作用是保证光源朝下
        d_list[2] = new flip_face(new xz_rect(light, 213, 343, 227, 332, 554));    // 光源 (150, 400, 150, 400, 554)
        d_list[3] = new xz_rect(white, 0, 555, 0, 555, 0);  // white
        d_list[4] = new flip_face(new xz_rect(white, 0, 555, 0, 555, 555)); // white
        d_list[5] = new flip_face(new xy_rect(white, 0, 555, 0, 555, 555)); // white

        // 先旋转再平移，否则无法得到正确的位置（原因：旋转轴是坐标轴y，所以需要将想作为旋转轴的线与坐标轴重合）
        hittable* box1 = new box(white, vec3(0, 0, 0), vec3(165, 165, 165));    /// (0,0,0) (165,165,165)
        box1 = new rotate_y(box1, -18);
        d_list[6] = new translate(box1, vec3(130, 0, 65));  //(130,0,65)
        //hittable *sphere = new sphere(aluminum, vec3(190, 90, 190), 90);
        //d_list[6] = new sphere(glass, vec3(190, 90, 190), 90);
        hittable* box2 = new box(white, vec3(0, 0, 0), vec3(165, 330, 165));      /// (0,0,0) (165,330,165)
        box2 = new rotate_y(box2, 15);
        d_list[7] = new translate(box2, vec3(265, 0.01, 295)); // (265,0,295)

        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 8);

        vec3 lookfrom(278, 278, -800);  // 278, 278, -800
        vec3 lookat(278 , 278, 0);
        float dist_to_focus = 1355; (lookfrom-lookat).length();
        float aperture = 0; //0.1
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 40,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus,
                                 0.0,
                                 1.0);
    }
}

__global__ void create_shape(hittable **shape, hittable **shape_list, int num_shape)
{
    shape[0] = new xz_rect(0, 213, 343, 227, 332, 554);
    //shape[1] = new sphere(0, vec3(190, 90, 190), 90);
    *shape_list = new hittable_list(shape, 1);
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera, hittable **shape)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        /*for (int i = 0; i < 8; ++i)
        {
            delete d_list[i]->mat_ptr;
            delete d_list[i];
        }*/
        delete *d_world;
        delete *d_camera;
        delete *shape;
    }
}

int main()
{
    const int nx = 1200;
    const int ny = 1200;
    const int ns = 100;     // 每个像素内样点数(抗锯齿)
    int tx = 16, ty = 16;

    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    cerr << "in " << tx << "x" << ty << " threads.\n";

    const int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged(&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMallocManaged(&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMallocManaged(&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 只能在host函数里调用stbi_load，读取图片
    int inx, iny, inn;
    int *d_inxyn;
    // string不能转换成const char*
    //string image_name = "IMG_20200910_000256.jpg"
    //"60847663_p0.jpg"
    unsigned char* texture_data = stbi_load("60847663_p0.jpg", &inx, &iny, &inn, 0);
    unsigned char* d_texture_data;
    // 复制图像
    checkCudaErrors(cudaMallocManaged(&d_texture_data, inx * iny * inn * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_texture_data, texture_data, inx * iny * inn * sizeof(unsigned char), cudaMemcpyHostToDevice));
    // 复制图像尺寸及通道数
    checkCudaErrors(cudaMallocManaged(&d_inxyn, 3 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_inxyn, &inx, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inxyn + 1, &iny, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inxyn + 2, &inn, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hittables & camera
    hittable **d_list;      
    hittable **d_world;     // 使用指针的指针的原因是，如果传递指针给creat_world则会传递副本，导致给临时量分配无效空间
    camera **d_camera;
    int num_hittables = 8;
    checkCudaErrors(cudaMallocManaged(&d_list, num_hittables * sizeof(hittable *)));
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(hittable *)));
    checkCudaErrors(cudaMallocManaged(&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, d_texture_data, d_inxyn);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    hittable **shape;
    hittable **shape_list;
    int num_shape = 1;
    checkCudaErrors(cudaMallocManaged(&shape, num_shape * sizeof(hittable *)));
    checkCudaErrors(cudaMallocManaged(&shape_list, sizeof(hittable *))); 
    create_shape<<<1, 1>>>(shape, shape_list, num_shape);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // render buffer
    dim3 blocks( (nx - 1) / tx + 1, (ny - 1) / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, shape_list, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    cerr << "took " << timer_seconds << " seconds to calculate.\n";

    // Output FB as Image
    cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; --j)
    {
        cerr << "\ralready writing: " << (int)(((double)(ny - j) / ny) * 100) << "%" << flush;
        for (int i = 0; i < nx; ++i)
        {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * clamp(fb[pixel_index].x(), 0.0, 0.999));
            int ig = int(255.99 * clamp(fb[pixel_index].y(), 0.0, 0.999));
            int ib = int(255.99 * clamp(fb[pixel_index].z(), 0.0, 0.999));
            cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list, d_world, d_camera, shape);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();

    cerr << "\nDone." << endl;
    return 0;
}