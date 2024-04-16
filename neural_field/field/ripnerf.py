from typing import Callable, Optional

import gin
import torch
from torch import Tensor, nn
import tinycudann as tcnn

from neural_field.encoding.rip_encoding import RipEncoding
from neural_field.nn_utils.activations import trunc_exp


@gin.configurable()
class RipNerf(nn.Module):
    def __init__(
        self,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
    ) -> None:
        super().__init__()
        self.encoding = RipEncoding()
        feature_dim = self.encoding.feature_dim
        self.geo_feat_dim = feature_dim - 1
        self.density_activation = density_activation

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out,
            n_output_dims=feature_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims
            + feature_dim
            - 1,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_color,
            },
        )

    def query_density(
        self,
        means: Tensor,  # [0, 1]
        covs: Optional[Tensor] = None,
        occ_res: Optional[Tensor] = None,
        return_feat: bool = False,
    ):
        means = means * 2.0 - 1.0  # [-1, 1]
        covs = covs * 4.0 if covs is not None else None
        in_sphere = torch.sum(means**2, dim=-1) <= 1.0

        enc = self.encoding(
            means=means[in_sphere, :],
            covs=covs[in_sphere, :, :] if covs is not None else None,
            occ_res=occ_res,
        )
        res = self.mlp_base(enc).to(means)
        density_before_activation, base_mlp_out = torch.split(
            res, [1, self.geo_feat_dim], dim=-1
        )
        density = torch.zeros(means.shape[0], 1).to(means)
        density[in_sphere, :] = self.density_activation(
            density_before_activation
        )

        if return_feat:
            feature = torch.zeros(means.shape[0], self.geo_feat_dim).to(means)
            feature[in_sphere, :] = base_mlp_out
        else:
            feature = None
        return {
            "density": density,
            "feature": feature,
        }

    def query_rgb(self, dir, embedding):
        # dir in [-1,1]
        dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.mlp_head(h).to(embedding)
        return {"rgb": rgb}
