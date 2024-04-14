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
        plane_distribution: str = "planotic_solid",
        n_vertices: int = 10,
        level_offset: float = 0.0,
        include_xyz: bool = False,
        learn_thetas: bool = False,
    ):
        super(RipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_res = plane_res
        self.feature_dim = feature_dim
        self.n_vertices = n_vertices
        self.level_offset = level_offset
        self.include_xyz = include_xyz
        self.learn_thetas = learn_thetas
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO: data, generate vertices and projection matrix, should we include them?
        self.projection_matrix_list = []
        if plane_distribution != "optimized":
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
                # vertices distributed in spherical white noise
                theta, phi = torch.rand(n_vertices) * math.pi, torch.rand(n_vertices) * 2 * math.pi
                vertices = torch.stack([
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.sin(phi),
                    torch.cos(theta)
                ], dim=-1)
            else:
                import numpy as np
                assert plane_distribution == "spherical_blue_noise" or plane_distribution == "golden_spiral"
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
        else:
            import numpy as np
            save_path = f"./data/rotation_matrix_{n_vertices}.npy"
            rotation_matrixs = np.load(save_path)
            for i in range(n_vertices):
                R = torch.tensor(rotation_matrixs[i], dtype=torch.float32).to(self.device)
                P = R[:2, :]
                self.projection_matrix_list.append(P)

        self.register_parameter(
            "fm",
            nn.Parameter(
                torch.zeros(n_vertices, plane_res, plane_res, feature_dim)
            ),
        )
        if learn_thetas:
            self.register_parameter(
                "thetas",
                nn.Parameter(torch.zeros(n_vertices)),
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
        if self.learn_thetas:
            nn.init.uniform_(self.thetas, -math.pi, math.pi)

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
        covs: Tensor = None,
        occ_res: Tensor = None,
    ):
        # means in [-1, 1]
        # means is Nx3, covs is Nx3x3
        self.update_ripmaps()
        input_list = []
        for i in range(self.n_vertices):
            P = self.projection_matrix_list[i]
            if self.learn_thetas:
                theta = self.thetas[i]
                R2D = torch.tensor(
                    [
                        [torch.cos(theta), -torch.sin(theta)],
                        [torch.sin(theta), torch.cos(theta)],
                    ]
                ).to(self.device)
                P = R2D @ P
            means_proj = means @ P.T
            if occ_res is not None:
                level = torch.empty_like(means_proj[..., :2]).fill_(-torch.log2(occ_res))
            else:
                covs_proj = P @ covs @ P.T
                sigmas = torch.sqrt(torch.diagonal(covs_proj, dim1=-2, dim2=-1))
                level = (
                    torch.log2(sigmas) + self.level_offset
                )
            level = level + torch.log2(self.plane_res)
            level = torch.clamp(level, 0, self.n_levels - 1)
            input_list.append(means_proj)
            input_list.append(level)
        input = torch.cat(input_list, dim=-1)
        enc = multi_anisomip_interp(self.ripmaps, input)

        if self.include_xyz:
            enc = torch.cat([means, enc], dim=-1)
        return enc
