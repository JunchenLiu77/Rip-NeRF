from typing import Union, List, Optional, Callable

import gin
import torch
import nerfacc
from nerfacc import render_weight_from_density, accumulate_along_rays

from neural_field.model.rf_model import RFModel
from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer
from neural_field.field.ripnerf import RipNerf


@gin.configurable()
class RipNerfModel(RFModel):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 1024,
        occ_res: int = 128,
        occ_thre: float = 5e-3,
    ) -> None:
        super().__init__(aabb=aabb, samples_per_ray=samples_per_ray)
        self.occ_thre = occ_thre
        self.field = RipNerf()
        self.ray_sampler = nerfacc.OccupancyGrid(
            roi_aabb=self.aabb, resolution=occ_res
        )

    def before_iter(self, step):
        # update_ray_sampler
        self.ray_sampler.every_n_step(
            step=step,
            occ_eval_fn=lambda x: self.field.query_density(
                means=self.contraction(x),
                occ_res=self.ray_sampler.resolution[0],
            )['density']
            * self.render_step_size,
            occ_thre=self.occ_thre,
        )

    @staticmethod
    def compute_ball_radii(distance, radii, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radii
        sample_ball_radii = distance * radii * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radii

    @staticmethod
    def compute_gaussian(origins, directions, starts, ends, radii):
        # Approximates conical frustums with a Gaussian distributions.
        # Source: https://github.com/nerfstudio-project/nerfstudio/blob/1f6fb8337c05afb357d61815416266a591966c61/nerfstudio/utils/math.py#L187

        mu = (starts + ends) / 2.0
        hw = (ends - starts) / 2.0

        means = origins + directions * (
            mu + (2.0 * mu * hw**2.0) / (3.0 * mu**2.0 + hw**2.0)
        )
        dir_variance = (hw**2) / 3 - (4 / 15) * (
            (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2
        )
        radii_variance = radii**2 * (
            (mu**2) / 4
            + (5 / 12) * hw**2
            - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2)
        )
        dir_outer_product = directions[..., :, None] * directions[..., None, :]
        eye = torch.eye(directions.shape[-1], device=directions.device)
        dir_mag_sq = torch.clamp(
            torch.sum(directions**2, dim=-1, keepdim=True), min=1e-10
        )
        null_outer_product = (
            eye
            - directions[..., :, None] * (directions / dir_mag_sq)[..., None, :]
        )
        dir_cov_diag = dir_variance[..., None] * dir_outer_product[..., :, :]
        radii_cov_diag = (
            radii_variance[..., None] * null_outer_product[..., :, :]
        )
        covs = dir_cov_diag + radii_cov_diag
        return means, covs

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
    ):
        # Ray sampling with occupancy grid
        with torch.no_grad():

            def sigma_fn(t_starts, t_ends, ray_indices):
                ray_indices = ray_indices.long()
                t_origins = rays.origins[ray_indices]
                t_dirs = rays.directions[ray_indices]
                radii = rays.radii[ray_indices]
                gs_means, gs_covs = self.compute_gaussian(
                    t_origins,
                    t_dirs,
                    t_starts,
                    t_ends,
                    radii,
                )
                gs_means_contracted, gs_covs_contracted = self.contraction(
                    gs_means, gs_covs
                )
                return self.field.query_density(
                    means=gs_means_contracted,
                    covs=gs_covs_contracted,
                )["density"]

            ray_indices, t_starts, t_ends = nerfacc.ray_marching(
                rays.origins,
                rays.directions,
                scene_aabb=self.aabb,
                grid=self.ray_sampler,
                sigma_fn=sigma_fn,
                render_step_size=self.render_step_size,
                stratified=self.training,
                early_stop_eps=1e-4,
            )

        # Ray rendering
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays.origins[ray_indices]
            t_dirs = rays.directions[ray_indices]
            radii = rays.radii[ray_indices]
            gs_means, gs_covs = self.compute_gaussian(
                t_origins,
                t_dirs,
                t_starts,
                t_ends,
                radii,
            )
            gs_means_contracted, gs_covs_contracted = self.contraction(
                gs_means, gs_covs
            )
            res = self.field.query_density(
                means=gs_means_contracted,
                covs=gs_covs_contracted,
                return_feat=True,
            )
            density, feature = res['density'], res['feature']
            rgb = self.field.query_rgb(dir=t_dirs, embedding=feature)['rgb']
            return rgb, density

        return self.rendering(
            t_starts,
            t_ends,
            ray_indices,
            rays,
            rgb_sigma_fn,
            render_bkgd=background_color,
        )

    def rendering(
        self,
        # ray marching results
        t_starts: torch.Tensor,
        t_ends: torch.Tensor,
        ray_indices: torch.Tensor,
        rays: RayBundle,
        # radiance field
        rgb_sigma_fn: Callable,  # rendering options
        render_bkgd: Optional[torch.Tensor] = None,
    ) -> RenderBuffer:
        n_rays = rays.origins.shape[0]
        # Query sigma/alpha and color with gradients
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())

        # Rendering
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        sample_buffer = {
            'num_samples': torch.as_tensor(
                [len(t_starts)], dtype=torch.int32, device=rgbs.device
            ),
        }

        # Rendering: accumulate rgbs, opacities, and depths along the rays.
        colors = accumulate_along_rays(
            weights, ray_indices=ray_indices, values=rgbs, n_rays=n_rays
        )
        opacities = accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        opacities.clamp_(
            0.0, 1.0
        )  # sometimes it may slightly bigger than 1.0, which will lead abnormal behaviours

        depths = accumulate_along_rays(
            weights,
            ray_indices=ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )
        depths = (
            depths * rays.ray_cos
        )  # from distance to real depth (z value in camera space)

        # Background composition.
        if render_bkgd is not None:
            colors = colors + render_bkgd * (1.0 - opacities)

        return RenderBuffer(
            rgb=colors,
            alpha=opacities,
            depth=depths,
            **sample_buffer,
            _static_field=set(sample_buffer),
        )

    @gin.configurable()
    def get_optimizer(
        self, lr=2e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        params_list = []
        params_list.append(
            dict(
                params=self.field.encoding.parameters(),
                lr=lr * feature_lr_scale,
            )
        )
        params_list.append(
            dict(params=self.field.direction_encoding.parameters(), lr=lr)
        )
        params_list.append(dict(params=self.field.mlp_base.parameters(), lr=lr))
        params_list.append(dict(params=self.field.mlp_head.parameters(), lr=lr))

        optim = torch.optim.AdamW(
            params_list,
            weight_decay=weight_decay,
            **kwargs,
            eps=1e-15,
        )
        return optim
