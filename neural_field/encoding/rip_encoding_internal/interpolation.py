from typing import Callable, List
import torch
from torch import Tensor
from .cuda import _C


class BilinearInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid_feats, points):
        feat_interp = _C.bilinear_fw(grid_feats, points)

        ctx.save_for_backward(grid_feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        grid_feats, points = ctx.saved_tensors

        dL_dfeats = _C.bilinear_bw(
            dL_dfeat_interp.contiguous(), grid_feats, points
        )

        return dL_dfeats, None


bilinear_interp = BilinearInterpolation.apply


class TriplaneInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        feat_interp = _C.triplane_fw(feats, points)

        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        feats, points = ctx.saved_tensors

        dL_dfeats = _C.triplane_bw(dL_dfeat_interp.contiguous(), feats, points)

        return dL_dfeats, None


triplane_interp = TriplaneInterpolation.apply


class TrimipInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, *mipmaps):
        feat_interp = _C.trimip_fw(mipmaps, points)

        ctx.save_for_backward(points, *mipmaps)

        return feat_interp

    # @staticmethod
    def backward(ctx, dL_dfeat_interp):
        points, *mipmaps = ctx.saved_tensors
        dL_dmipmaps = _C.trimip_bw(
            dL_dfeat_interp.contiguous(), mipmaps, points
        )

        return None, *dL_dmipmaps


trimip_interp = lambda mipmaps, points: TrimipInterpolation.apply(
    points, *mipmaps
)


class TriAnisomipInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, *mipmaps):
        feat_interp = _C.tri_anisomip_fw(mipmaps, points)

        ctx.save_for_backward(points, *mipmaps)

        return feat_interp

    # @staticmethod
    def backward(ctx, dL_dfeat_interp):
        points, *mipmaps = ctx.saved_tensors
        dL_dmipmaps = _C.tri_anisomip_bw(
            dL_dfeat_interp.contiguous(), mipmaps, points
        )

        return None, *dL_dmipmaps


tri_anisomip_interp = lambda mipmaps, points: TriAnisomipInterpolation.apply(
    points, *mipmaps
)


class MultiAnisomipInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, *mipmaps):
        feat_interp = _C.multi_anisomip_fw(mipmaps, points)

        ctx.save_for_backward(points, *mipmaps)

        return feat_interp

    # @staticmethod
    def backward(ctx, dL_dfeat_interp):
        points, *mipmaps = ctx.saved_tensors
        dL_dmipmaps = _C.multi_anisomip_bw(
            dL_dfeat_interp.contiguous(), mipmaps, points
        )

        return None, *dL_dmipmaps


multi_anisomip_interp: Callable[[List, Tensor], Tensor] = (
    lambda mipmaps, points: MultiAnisomipInterpolation.apply(points, *mipmaps)
)
