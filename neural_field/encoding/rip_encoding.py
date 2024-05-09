from typing import Optional, Literal
import torch
import math
import gin
from torch import Tensor, nn
from .rip_encoding_internal import multi_anisomip_interp


@gin.configurable()
class RipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int = 8,
        plane_res: int = 512,
        feature_dim: int = 16,
        plane_distribution: Literal[
            "planotic_solid",
            "spherical_white_noise",
            "spherical_blue_noise",
            "golden_spiral",
        ] = "planotic_solid",
        n_vertices: int = 10,
        scale_factor: float = 2.0,
        include_xyz: bool = False,
    ):
        super(RipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_res = plane_res
        self.log2_phane_res = torch.log2(
            torch.tensor(plane_res, dtype=torch.float32)
        )
        self.feature_dim = feature_dim
        self.n_vertices = n_vertices
        self.scale_factor = scale_factor
        self.include_xyz = include_xyz
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.projection_matrix_list = []
        if plane_distribution == "planotic_solid":
            # vertices of regular polyhedron
            if n_vertices == 3:
                # cube
                vertices = torch.tensor(
                    [
                        [1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0],
                    ]
                )
            elif n_vertices == 4:
                # tetrahedron
                # r2 = math.sqrt(2)
                # vertices = torch.tensor(
                #     [
                #         [r2, 0, -1.0],
                #         [-r2, 0, -1.0],
                #         [0, r2, 1.0],
                #         [0, -r2, 1.0],
                #     ]
                # )
                # octahedron
                vertices = torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, -1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, -1.0, -1.0],
                    ]
                )
            elif n_vertices == 6:
                # icosahedron
                phi = (1 + math.sqrt(5)) / 2
                vertices = torch.tensor(
                    [
                        [0, 1.0, phi],
                        [0, -1.0, phi],
                        [1.0, phi, 0],
                        [-1.0, phi, 0],
                        [phi, 0, 1.0],
                        [phi, 0, -1.0],
                    ]
                )
            elif n_vertices == 10:
                # dodecahedron
                phi = (1 + math.sqrt(5)) / 2
                vertices = torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, -1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, -1.0, -1.0],
                        [0, phi, 1.0 / phi],
                        [0, phi, -1.0 / phi],
                        [1.0 / phi, 0, phi],
                        [1.0 / phi, 0, -phi],
                        [phi, 1.0 / phi, 0],
                        [phi, -1.0 / phi, 0],
                    ]
                )
            else:
                raise NotImplementedError
        elif plane_distribution == "spherical_white_noise":
            theta, phi = (
                torch.rand(n_vertices) * math.pi,
                torch.rand(n_vertices) * 2 * math.pi,
            )
            vertices = torch.stack(
                [
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta),
                ],
                dim=-1,
            )
        else:
            import numpy as np

            assert (
                plane_distribution == "spherical_blue_noise"
                or plane_distribution == "golden_spiral"
            )
            save_path = f"./data/{plane_distribution}_{n_vertices}.npy"
            vertices = torch.tensor(np.load(save_path), dtype=torch.float32)
            assert vertices.shape[0] == n_vertices

        self.vertices = (
            vertices / torch.linalg.norm(vertices, dim=-1, keepdim=True)
        ).to(self.device)

        for i in range(n_vertices):
            axis = self.vertices[i]
            # find any set of standard orthogonal bases
            if axis[0] != 0 or axis[1] != 0:
                p0 = torch.tensor([-axis[1], axis[0], 0.0]).to(self.device)
            else:
                p0 = torch.tensor([0.0, -axis[2], axis[1]]).to(self.device)
            p1 = torch.cross(axis, p0)
            p0 /= torch.norm(p0)
            p1 /= torch.norm(p1)
            P = torch.stack([p0, p1], dim=0)
            self.projection_matrix_list.append(P)

        self.register_parameter(
            "fm",
            nn.Parameter(
                torch.zeros(n_vertices, plane_res, plane_res, feature_dim)
            ),
        )
        self.init_parameters()
        self.dim_out = self.feature_dim * self.n_vertices + (
            3 if include_xyz else 0
        )
        self.avg_pool1x2 = nn.AvgPool2d(kernel_size=(1, 2)).to(self.device)
        self.avg_pool2x1 = nn.AvgPool2d(kernel_size=(2, 1)).to(self.device)

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def update_ripmaps(self) -> None:
        ripmaps = []
        for l1 in range(self.n_levels):
            for l2 in range(self.n_levels):
                if l1 == 0 and l2 == 0:
                    ripmap = self.fm
                elif l2 == 0:
                    ripmap = self.avg_pool1x2(
                        ripmaps[(l1 - 1) * self.n_levels].permute(0, 3, 1, 2)
                    ).permute(0, 2, 3, 1)
                else:
                    ripmap = self.avg_pool2x1(
                        ripmaps[-1].permute(0, 3, 1, 2)
                    ).permute(0, 2, 3, 1)
                ripmaps.append(ripmap.clone())
        self.ripmaps = ripmaps

    def forward(
        self,
        means: Tensor,
        covs: Optional[Tensor] = None,
        occ_res: Optional[Tensor] = None,
    ):
        # means in [-1, 1]
        # means is Nx3, covs is Nx3x3
        self.update_ripmaps()
        input_list = []
        for i in range(self.n_vertices):
            P = self.projection_matrix_list[i]
            means_proj = means @ P.T
            if occ_res is not None:
                level = torch.empty_like(means_proj[..., :2]).fill_(
                    -torch.log2(occ_res)
                )
            else:
                covs_proj = P @ covs @ P.T
                sigmas = torch.sqrt(torch.diagonal(covs_proj, dim1=-2, dim2=-1))
                level = torch.log2(sigmas * self.scale_factor)
            level = level + self.log2_phane_res
            level = torch.clamp(level, 0, self.n_levels - 1)
            input_list.append(means_proj)
            input_list.append(level)
        input = torch.cat(input_list, dim=-1)
        enc = multi_anisomip_interp(self.ripmaps, input)

        if self.include_xyz:
            enc = torch.cat([means, enc], dim=-1)
        return enc