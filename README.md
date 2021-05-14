# ray-tracing-rest-life

this is a ray-tracing program using CUDA to accelerate up. Modified from Peter Shirly's [ONE-WEEK series](https://github.com/RayTracing/raytracing.github.io).

# keywords
* CUDA
* Mental, Diffuse Materials, Dielectrics.
* Motion blue
* Defocus blur
* Texture
* Monte Carlo 
* ONB class
* Sampling the Light 

Follow the origin implementation, I didn't use shadow-ray but use another method that sampling the light with more probability.

The pity is that I haven't implement the BVH. I found is diffcult to implement in CUDA, but I still get a decent and fast ray-tracing engine that can easily 
produce some complicated scenes.

You can consult the books in the link above for any concepts about ray-tracing.

the pic below is my rendering result. A cornell box whit 1000 ray in one pixel.
