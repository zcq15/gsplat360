
# gsplat360

A gsplat-based rasterization designed for panoramic cameras, with support for 360-degree rendering for both 3DGS and 2DGS.

This repository is derived from [gsplat v1.4.0](https://github.com/nerfstudio-project/gsplat/tree/v1.4.0).  
We gratefully acknowledge the original authors for their open-source contribution.

This project includes modifications to the original codebase to enable panoramic image rendering for 3D Gaussian Splatting (3DGS) and 2D Gaussian Splatting (2DGS).

This project is distributed under the Apache License 2.0.  
Please see the [LICENSE](./LICENSE) file for details.

## Installation

### Dependencies

Please install [PyTorch](https://pytorch.org/get-started/locally/) first.

### Install from Source

Clone the repository and install the package:

```bash
git clone --recursive git@github.com:zcq15/gsplat360.git
cd gsplat360
pip install .
```

## Examples

The following examples illustrate the main panoramic rendering interfaces provided by this project.

**Panoramic 3DGS**
```python

render, alpha, render_distort, info = rasterization(
    # Most parameters follow those of the corresponding function in gsplat.
    means: Tensor,  # [N, 3]
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    covars: Optional[Tensor] = None,

    # Enable panoramic rendering.
    camera_model: Literal["pinhole", "ortho", "fisheye", "equirectangular"] = "equirectangular",
    
    # ====== Optional ======

    # Enable depth distortion loss from 2DGS.
    distloss: bool = False,

    # Get the occurrence count N_g and cumulative weight W_g corresponding to the maximum response.
    # i.e., N_g = |Ind_g|, W_g = \sum_p \alpha_p, where p \in Ind_g,
    # and Ind_g is the set of pixels dominated by Gaussian g.
    ret_visible: bool = False, # info["accum_times"] -> N_g, [C, N]; info["accum_visible"] -> W_g, [C, N]

    # Get the weighted sum of Gaussian features at the pixel with the maximum response.
    # i.e., f_g = \sum_p \alpha_p f_p, where p \in Ind_g.
    query_values: Optional[Tensor] = None,   # [C, image_height, image_width, D], info["query_answers"] -> f_g, [C, D]

    # Gaussian confidence is introduced for backpropagating gradients to the Gaussian-to-view matrix.
    # When set to None or to an all-one tensor, the behavior is equivalent to the standard chain-rule implementation.
    gauss_confs: Optional[Tensor] = None,  # [N]
)

```

**Panoramic 2DGS**
``` python
# The parameter definitions are the same as those of equir_rasterization in gsplat.
( render_colors,
  render_alphas,
  render_normals,
  render_normals_from_depth,
  render_distort,
  render_median,
  meta) =  equir_rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
  )
```

## Citation

If you find this project useful in your research or applications, please consider citing:


```
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting}, 
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765}, 
}

@misc{zhuang2026posefreeomnidirectionalgaussiansplatting,
      title={Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors}, 
      author={Chuanqing Zhuang and Xin Lu and Zehui Deng and Zhengda Lu and Yiqun Wang and Junqi Diao and Jun Xiao},
      year={2026},
      eprint={2603.23324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.23324}, 
}

```