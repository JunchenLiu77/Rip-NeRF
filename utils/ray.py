import torch
from dataclasses import dataclass, field
from typing import Optional

from utils.tensor_dataclass import TensorDataclass


@dataclass
class RayBundle(TensorDataclass):
    origins: torch.Tensor = field(default_factory=lambda: torch.Tensor())
    """Ray origins (XYZ)"""

    directions: torch.Tensor = field(default_factory=lambda: torch.Tensor())
    """Unit ray direction vector"""

    radii: torch.Tensor = field(default_factory=lambda: torch.Tensor())
    """Ray image plane intersection circle radii"""

    ray_cos: torch.Tensor = field(default_factory=lambda: torch.Tensor())
    """Ray cos"""

    def __len__(self):
        num_rays = torch.numel(self.origins) // self.origins.shape[-1]
        return num_rays

    @property
    def shape(self):
        return list(super().shape)


@dataclass
class RayBundleExt(RayBundle):
    ray_depth: Optional[torch.Tensor] = None


@dataclass
class RayBundleRast(RayBundleExt):
    ray_uv: Optional[torch.Tensor] = None
    ray_mip_level: Optional[torch.Tensor] = None
