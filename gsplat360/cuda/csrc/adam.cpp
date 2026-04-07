// Based on gsplat; modified from the original.
// Licensed under the Apache License, Version 2.0.
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "adam.h"     // where the launch function is declared
#include "bindings.h" // where all the macros are defined

namespace gsplat {

void adam(
    at::Tensor &param,                    // [N, ...]
    const at::Tensor &param_grad,         // [N, ...]
    at::Tensor &exp_avg,                  // [N, ...]
    at::Tensor &exp_avg_sq,               // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    at::Tensor &step_per_param,           // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps) {
    GSPLAT_DEVICE_GUARD(param);
    GSPLAT_CHECK_INPUT(param);
    GSPLAT_CHECK_INPUT(param_grad);
    GSPLAT_CHECK_INPUT(exp_avg);
    GSPLAT_CHECK_INPUT(exp_avg_sq);
    GSPLAT_CHECK_INPUT(step_per_param);
    if (valid.has_value()) {
        GSPLAT_CHECK_INPUT(valid.value());
        TORCH_CHECK(valid.value().dim() == 1, "valid should be 1D tensor");
        TORCH_CHECK(
            valid.value().size(0) == param.size(0),
            "valid first dimension should match param first dimension");
    }

    launch_adam_kernel(
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        valid,
        step_per_param,
        lr,
        b1,
        b2,
        eps);
}

} // namespace gsplat
