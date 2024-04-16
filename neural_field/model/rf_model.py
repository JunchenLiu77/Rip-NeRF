import math
from typing import Union, List, Dict

import gin
import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.ray import RayBundle
from utils.render_buffer import RenderBuffer


# @gin.configurable()
class RFModel(nn.Module):
    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        samples_per_ray: int = 1024,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.samples_per_ray = samples_per_ray
        self.render_step_size = (
            (self.aabb[3:] - self.aabb[:3]).max()
            * math.sqrt(3)
            / samples_per_ray
        ).item()
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        self.aabb_size = aabb_max - aabb_min
        assert (
            self.aabb_size[0] == self.aabb_size[1] == self.aabb_size[2]
        ), "Current implementation only supports cube aabb"
        self.field = None
        self.ray_sampler = None

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg', normalize=True
        )

    def contraction(self, means, covs=None):
        # contract means to [0, 1]
        r = self.aabb_size[0] / 2.0
        means = (means + r) / (2 * r)
        if covs == None:
            return means
        else:
            covs = covs / (4 * r**2)
            return means, covs

    def before_iter(self, step):
        pass

    def after_iter(self, step):
        pass

    def forward(
        self,
        rays: RayBundle,
        background_color=None,
    ):
        raise NotImplementedError

    @gin.configurable()
    def get_optimizer(
        self, lr=1e-3, weight_decay=1e-5, feature_lr_scale=10.0, **kwargs
    ):
        raise NotImplementedError

    @gin.configurable()
    def compute_loss(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        metric='smooth_l1',
        **kwargs
    ) -> Dict:
        if 'smooth_l1' == metric:
            loss_fn = F.smooth_l1_loss
        elif 'mse' == metric:
            loss_fn = F.mse_loss
        elif 'mae' == metric:
            loss_fn = F.l1_loss
        else:
            raise NotImplementedError

        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        loss = loss_fn(
            rb.rgb[alive_ray_mask], target.rgb[alive_ray_mask], reduction='none'
        )
        loss = (
            loss * target.loss_multi[alive_ray_mask]
        ).sum() / target.loss_multi[alive_ray_mask].sum()
        return {'total_loss': loss}

    @gin.configurable()
    def compute_metrics(
        self,
        rays: RayBundle,
        rb: RenderBuffer,
        target: RenderBuffer,
        # Configurable
        **kwargs
    ) -> Dict:
        # ray info
        alive_ray_mask = (rb.alpha.squeeze(-1) > 0).detach()
        rendering_samples_actual = rb.num_samples[0].item()
        ray_info = {
            'num_alive_ray': alive_ray_mask.long().sum().item(),
            'rendering_samples_actual': rendering_samples_actual,
            'num_rays': len(target),
        }
        # quality
        if len(rb.rgb.shape) == 2:
            quality = {
                'PSNR': self.psnr(rb.rgb, target.rgb).item(),
            }
        else:
            rgb_clipped = torch.clip(rb.rgb, min=0, max=1)
            target_clipped = torch.clip(target.rgb, min=0, max=1)
            quality = {
                'PSNR': self.psnr(rgb_clipped, target_clipped).item(),
                'SSIM': self.ssim(
                    rgb_clipped.permute(2, 0, 1).unsqueeze(0),
                    target_clipped.permute(2, 0, 1).unsqueeze(0),
                ).item(),
                'LPIPS': self.lpips(
                    rgb_clipped.permute(2, 0, 1).unsqueeze(0),
                    target_clipped.permute(2, 0, 1).unsqueeze(0),
                ).item(),
            }
        return {**ray_info, **quality}
