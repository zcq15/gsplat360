// Based on gsplat; modified from the original.
// Licensed under the Apache License, Version 2.0.

#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Equirectangular Projection of Gaussians (Single Batch) Forward Pass 2DGS
 ****************************************************************************/

template <typename T>
__global__ void equir_fully_fused_projection_fwd_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const T* __restrict__ means, // [N, 3]
    const T* __restrict__ quats, // [N, 4]
    const T* __restrict__ scales, // [N, 3]
    const T* __restrict__ viewmats, // [C, 4, 4]
    const T* __restrict__ Ks, // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    // outputs
    int32_t* __restrict__ radii, // [C, N]
    T* __restrict__ means2d, // [C, N, 2]
    T* __restrict__ depths, // [C, N]
    T* __restrict__ ray_transforms, // [C, N, 3, 3]
    T* __restrict__ normals // [C, N, 3]
)
{
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);

    T depth = sqrtf(mean_c.x * mean_c.x + mean_c.y * mean_c.y + mean_c.z * mean_c.z);
    if (depth < near_plane || depth > far_plane) {
        radii[idx] = 0;
        return;
    }

    // build ray transformation matrix and transform from world space to camera
    // space
    quats += gid * 4;
    scales += gid * 3;

    mat3<T> RS_camera = R * quat_to_rotmat<T>(glm::make_vec4(quats)) * mat3<T>(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);

    mat3<T> WH = mat3<T>(RS_camera[0], RS_camera[1], mean_c);

    // normals dual visible
    vec3<T> normal = RS_camera[2];
    T multipler = glm::dot(-normal, mean_c) > 0 ? 1 : -1;
    normal *= multipler;

    if (abs(glm::dot(normal, mean_c)) <= 1e-3f * glm::length(mean_c)) {
        radii[idx] = 0;
        return;
    };

    vec2<T> mean2d;
    T radius;
    mat3<T> RS_camera_2d = R * quat_to_rotmat<T>(glm::make_vec4(quats)) * mat3<T>(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 0.0);
    mat3<T> covar_c = RS_camera_2d * glm::transpose(RS_camera_2d);
    mat2<T> covar2d;
    equir_proj<T>(mean_c, covar_c, image_width, image_height, covar2d, mean2d);
    tangent_proj<T>(mean_c, covar_c, image_width, image_height, covar2d, mean2d);
    T compensation;
    T det = add_blur(0.3f, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }
    // take 3 sigma as the radius (non differentiable)
    T b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    T v1 = b + sqrtf(max(0.01f, b * b - det));
    radius = ceil(3.f * sqrtf(v1));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = depth;
    ray_transforms[idx * 9] = WH[0][0];
    ray_transforms[idx * 9 + 1] = WH[0][1];
    ray_transforms[idx * 9 + 2] = WH[0][2];
    ray_transforms[idx * 9 + 3] = WH[1][0];
    ray_transforms[idx * 9 + 4] = WH[1][1];
    ray_transforms[idx * 9 + 5] = WH[1][2];
    ray_transforms[idx * 9 + 6] = WH[2][0];
    ray_transforms[idx * 9 + 7] = WH[2][1];
    ray_transforms[idx * 9 + 8] = WH[2][2];
    normals[idx * 3] = normal.x;
    normals[idx * 3 + 1] = normal.y;
    normals[idx * 3 + 2] = normal.z;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
equir_fully_fused_projection_fwd_2dgs_tensor(
    const torch::Tensor& means, // [N, 3]
    const torch::Tensor& quats, // [N, 4]
    const torch::Tensor& scales, // [N, 3]
    const torch::Tensor& viewmats, // [C, 4, 4]
    const torch::Tensor& Ks, // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip)
{
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0); // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({ C, N }, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({ C, N, 2 }, means.options());
    torch::Tensor depths = torch::empty({ C, N }, means.options());
    torch::Tensor ray_transforms = torch::empty({ C, N, 3, 3 }, means.options());
    torch::Tensor normals = torch::empty({ C, N, 3 }, means.options());

    if (C && N) {
        equir_fully_fused_projection_fwd_2dgs_kernel<float>
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                quats.data_ptr<float>(),
                scales.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                near_plane,
                far_plane,
                radius_clip,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                normals.data_ptr<float>());
    }
    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

} // namespace gsplat