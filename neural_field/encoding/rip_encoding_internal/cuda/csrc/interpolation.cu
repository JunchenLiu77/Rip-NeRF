#include <torch/extension.h>
#include "include/util.h"

template <typename scalar_t>
__global__ void bilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grid_feat,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int height = grid_feat.size(0),
              width = grid_feat.size(1),
              channel_num = grid_feat.size(2);

    if (n >= points.size(0) || c >= channel_num)
        return;

    // point [-1,1]
    const InterpCoord x = quantize_coord(points[n][0], width),
                      y = quantize_coord(points[n][1], height);

    const scalar_t w1 = (1 - x.distance) * (1 - y.distance),
                   w2 = (1 - x.distance) * y.distance,
                   w3 = x.distance * (1 - y.distance),
                   w4 = 1 - w1 - w2 - w3;
    feat_interp[n][c] = w1 * grid_feat[y.low][x.low][c] +
                        w2 * grid_feat[y.high][x.low][c] +
                        w3 * grid_feat[y.low][x.high][c] +
                        w4 * grid_feat[y.high][x.high][c];
}

template <typename scalar_t>
__global__ void bilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grid_feat,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dgrid_feats,
    const int height,
    const int width)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= points.size(0) || c >= grid_feat.size(2))
        return;

    // point [-1,1]
    const InterpCoord x = quantize_coord(points[n][0], width),
                      y = quantize_coord(points[n][1], height);

    const scalar_t w1 = (1 - x.distance) * (1 - y.distance),
                   w2 = (1 - x.distance) * y.distance,
                   w3 = x.distance * (1 - y.distance),
                   w4 = 1 - w1 - w2 - w3;

    atomicAdd(&dL_dgrid_feats[y.low][x.low][c], w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dgrid_feats[y.high][x.low][c], w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dgrid_feats[y.low][x.high][c], w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dgrid_feats[y.high][x.high][c], w4 * dL_dfeat_interp[n][c]);
}

torch::Tensor bilinear_fw(
    const torch::Tensor grid_feats,
    const torch::Tensor points)
{
    CHECK_INPUT(grid_feats);
    CHECK_INPUT(points);

    const int N = points.size(0), C = grid_feats.size(2);

    torch::Tensor feat_interp = torch::empty({N, C}, grid_feats.options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(grid_feats.type(), "bilinear_fw",
                               ([&]
                                { bilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
                                      grid_feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return feat_interp;
}

torch::Tensor bilinear_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor grid_feats,
    const torch::Tensor points)
{
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(grid_feats);
    CHECK_INPUT(points);

    const int N = points.size(0), H = grid_feats.size(0), W = grid_feats.size(1), C = grid_feats.size(2);

    torch::Tensor dL_dgrid_feats = torch::zeros({H, W, C}, grid_feats.options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(grid_feats.type(), "bilinear_bw",
                               ([&]
                                { bilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
                                      dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      grid_feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      dL_dgrid_feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                      H,
                                      W); }));

    return dL_dgrid_feats;
}

template <typename scalar_t>
__global__ void triplane_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feats,  // 3xHxWxC
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points, // Nx3
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp)  // Nx3C
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int height = feats.size(1),
              width = feats.size(2),
              channel_num = feats.size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    const InterpCoord x = quantize_coord(points[n][0], width),
                      y = quantize_coord(points[n][1], height),
                      y_w = quantize_coord(points[n][1], width),
                      z = quantize_coord(points[n][2], height);
    // palne XY
    scalar_t w1 = (1 - x.distance) * (1 - y.distance),
             w2 = (1 - x.distance) * y.distance,
             w3 = x.distance * (1 - y.distance),
             w4 = 1 - w1 - w2 - w3;
    feat_interp[n][c] = w1 * feats[0][y.low][x.low][c] +
                        w2 * feats[0][y.high][x.low][c] +
                        w3 * feats[0][y.low][x.high][c] +
                        w4 * feats[0][y.high][x.high][c];
    // palne XZ
    w1 = (1 - x.distance) * (1 - z.distance);
    w2 = (1 - x.distance) * z.distance;
    w3 = x.distance * (1 - z.distance);
    w4 = 1 - w1 - w2 - w3;
    feat_interp[n][c + channel_num] = w1 * feats[1][z.low][x.low][c] +
                                      w2 * feats[1][z.high][x.low][c] +
                                      w3 * feats[1][z.low][x.high][c] +
                                      w4 * feats[1][z.high][x.high][c];
    // palne YZ
    w1 = (1 - y_w.distance) * (1 - z.distance);
    w2 = (1 - y_w.distance) * z.distance;
    w3 = y_w.distance * (1 - z.distance);
    w4 = 1 - w1 - w2 - w3;
    feat_interp[n][c + channel_num * 2] = w1 * feats[2][z.low][y_w.low][c] +
                                          w2 * feats[2][z.high][y_w.low][c] +
                                          w3 * feats[2][z.low][y_w.high][c] +
                                          w4 * feats[2][z.high][y_w.high][c];
}

template <typename scalar_t>
__global__ void triplane_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dL_dfeats)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int height = feats.size(1),
              width = feats.size(2),
              channel_num = feats.size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    const InterpCoord x = quantize_coord(points[n][0], width),
                      y = quantize_coord(points[n][1], height),
                      y_w = quantize_coord(points[n][1], width),
                      z = quantize_coord(points[n][2], height);
    // palne XY
    scalar_t w1 = (1 - x.distance) * (1 - y.distance),
             w2 = (1 - x.distance) * y.distance,
             w3 = x.distance * (1 - y.distance),
             w4 = 1 - w1 - w2 - w3;
    atomicAdd(&dL_dfeats[0][y.low][x.low][c], w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dfeats[0][y.high][x.low][c], w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dfeats[0][y.low][x.high][c], w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dfeats[0][y.high][x.high][c], w4 * dL_dfeat_interp[n][c]);

    // palne XZ
    w1 = (1 - x.distance) * (1 - z.distance);
    w2 = (1 - x.distance) * z.distance;
    w3 = x.distance * (1 - z.distance);
    w4 = 1 - w1 - w2 - w3;
    atomicAdd(&dL_dfeats[1][z.low][x.low][c], w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dfeats[1][z.high][x.low][c], w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dfeats[1][z.low][x.high][c], w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dfeats[1][z.high][x.high][c], w4 * dL_dfeat_interp[n][c + channel_num]);

    // // palne YZ
    w1 = (1 - y_w.distance) * (1 - z.distance);
    w2 = (1 - y_w.distance) * z.distance;
    w3 = y_w.distance * (1 - z.distance);
    w4 = 1 - w1 - w2 - w3;
    atomicAdd(&dL_dfeats[2][z.low][y_w.low][c], w1 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dfeats[2][z.high][y_w.low][c], w2 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dfeats[2][z.low][y_w.high][c], w3 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dfeats[2][z.high][y_w.high][c], w4 * dL_dfeat_interp[n][c + channel_num * 2]);
}

torch::Tensor triplane_fw(
    const torch::Tensor feats,
    const torch::Tensor points)
{
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    const int N = points.size(0), C = feats.size(3);

    torch::Tensor feat_interp = torch::empty({N, C * 3}, feats.options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "triplane_fw",
                               ([&]
                                { triplane_fw_kernel<scalar_t><<<blocks, threads>>>(
                                      feats.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    return feat_interp;
}

torch::Tensor triplane_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points)
{
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    const int N = points.size(0), C = feats.size(3);

    torch::Tensor dL_dfeats = torch::zeros({feats.size(0), feats.size(1), feats.size(2), feats.size(3)}, feats.options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "triplane_bw",
                               ([&]
                                { triplane_bw_kernel<scalar_t><<<blocks, threads>>>(
                                      dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      feats.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                      points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                      dL_dfeats.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()); }));

    return dL_dfeats;
}

// template <typename scalar_t, typename scalar_t2>
__global__ void trimip_fw_kernel(
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (3xHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx4 (xyz+level)
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> feat_interp)  // Nx3C
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    const int level_floor = floor(points[n][3]);
    InterpCoord level = {};
    level.low = max(int(0), level_floor);
    level.high = min(int(level_num - 1), level_floor + 1);
    level.distance = points[n][3] - level_floor;

    const InterpCoord low_x = quantize_coord(points[n][0], mipmaps[level.low].size(2)),
                      high_x = quantize_coord(points[n][0], mipmaps[level.high].size(2)),
                      low_y = quantize_coord(points[n][1], mipmaps[level.low].size(1)),
                      high_y = quantize_coord(points[n][1], mipmaps[level.high].size(1)),
                      low_y_w = quantize_coord(points[n][1], mipmaps[level.low].size(2)),
                      high_y_w = quantize_coord(points[n][1], mipmaps[level.high].size(2)),
                      low_z = quantize_coord(points[n][2], mipmaps[level.low].size(1)),
                      high_z = quantize_coord(points[n][2], mipmaps[level.high].size(1));
    // palne XY
    float low_w1 = (1 - low_x.distance) * (1 - low_y.distance),
          low_w2 = (1 - low_x.distance) * low_y.distance,
          low_w3 = low_x.distance * (1 - low_y.distance),
          low_w4 = 1 - low_w1 - low_w2 - low_w3,
          high_w1 = (1 - high_x.distance) * (1 - high_y.distance),
          high_w2 = (1 - high_x.distance) * high_y.distance,
          high_w3 = high_x.distance * (1 - high_y.distance),
          high_w4 = 1 - high_w1 - high_w2 - high_w3;

    feat_interp[n][c] = (1. - level.distance) * (low_w1 * mipmaps[level.low][0][low_y.low][low_x.low][c] +
                                                 low_w2 * mipmaps[level.low][0][low_y.high][low_x.low][c] +
                                                 low_w3 * mipmaps[level.low][0][low_y.low][low_x.high][c] +
                                                 low_w4 * mipmaps[level.low][0][low_y.high][low_x.high][c]) +
                        level.distance * (high_w1 * mipmaps[level.high][0][high_y.low][high_x.low][c] +
                                          high_w2 * mipmaps[level.high][0][high_y.high][high_x.low][c] +
                                          high_w3 * mipmaps[level.high][0][high_y.low][high_x.high][c] +
                                          high_w4 * mipmaps[level.high][0][high_y.high][high_x.high][c]);
    // palne XZ
    low_w1 = (1 - low_x.distance) * (1 - low_z.distance),
    low_w2 = (1 - low_x.distance) * low_z.distance,
    low_w3 = low_x.distance * (1 - low_z.distance),
    low_w4 = 1 - low_w1 - low_w2 - low_w3,
    high_w1 = (1 - high_x.distance) * (1 - high_z.distance),
    high_w2 = (1 - high_x.distance) * high_z.distance,
    high_w3 = high_x.distance * (1 - high_z.distance),
    high_w4 = 1 - high_w1 - high_w2 - high_w3;

    feat_interp[n][c + channel_num] = (1. - level.distance) * (low_w1 * mipmaps[level.low][1][low_z.low][low_x.low][c] +
                                                               low_w2 * mipmaps[level.low][1][low_z.high][low_x.low][c] +
                                                               low_w3 * mipmaps[level.low][1][low_z.low][low_x.high][c] +
                                                               low_w4 * mipmaps[level.low][1][low_z.high][low_x.high][c]) +
                                      level.distance * (high_w1 * mipmaps[level.high][1][high_z.low][high_x.low][c] +
                                                        high_w2 * mipmaps[level.high][1][high_z.high][high_x.low][c] +
                                                        high_w3 * mipmaps[level.high][1][high_z.low][high_x.high][c] +
                                                        high_w4 * mipmaps[level.high][1][high_z.high][high_x.high][c]);

    // palne YZ
    low_w1 = (1 - low_y_w.distance) * (1 - low_z.distance),
    low_w2 = (1 - low_y_w.distance) * low_z.distance,
    low_w3 = low_y_w.distance * (1 - low_z.distance),
    low_w4 = 1 - low_w1 - low_w2 - low_w3,
    high_w1 = (1 - high_y_w.distance) * (1 - high_z.distance),
    high_w2 = (1 - high_y_w.distance) * high_z.distance,
    high_w3 = high_y_w.distance * (1 - high_z.distance),
    high_w4 = 1 - high_w1 - high_w2 - high_w3;

    feat_interp[n][c + channel_num * 2] = (1. - level.distance) * (low_w1 * mipmaps[level.low][2][low_z.low][low_y_w.low][c] +
                                                                   low_w2 * mipmaps[level.low][2][low_z.high][low_y_w.low][c] +
                                                                   low_w3 * mipmaps[level.low][2][low_z.low][low_y_w.high][c] +
                                                                   low_w4 * mipmaps[level.low][2][low_z.high][low_y_w.high][c]) +
                                          level.distance * (high_w1 * mipmaps[level.high][2][high_z.low][high_y_w.low][c] +
                                                            high_w2 * mipmaps[level.high][2][high_z.high][high_y_w.low][c] +
                                                            high_w3 * mipmaps[level.high][2][high_z.low][high_y_w.high][c] +
                                                            high_w4 * mipmaps[level.high][2][high_z.high][high_y_w.high][c]);
}

// template <typename scalar_t>
__global__ void trimip_bw_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (3xHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx4 (xyz+level)
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *dL_dmipmaps)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    const int level_floor = floor(points[n][3]);
    InterpCoord level = {};
    level.low = max(int(0), level_floor);
    level.high = min(int(level_num - 1), level_floor + 1);
    level.distance = points[n][3] - level_floor;

    const InterpCoord low_x = quantize_coord(points[n][0], mipmaps[level.low].size(2)),
                      high_x = quantize_coord(points[n][0], mipmaps[level.high].size(2)),
                      low_y = quantize_coord(points[n][1], mipmaps[level.low].size(1)),
                      high_y = quantize_coord(points[n][1], mipmaps[level.high].size(1)),
                      low_y_w = quantize_coord(points[n][1], mipmaps[level.low].size(2)),
                      high_y_w = quantize_coord(points[n][1], mipmaps[level.high].size(2)),
                      low_z = quantize_coord(points[n][2], mipmaps[level.low].size(1)),
                      high_z = quantize_coord(points[n][2], mipmaps[level.high].size(1));
    // palne XY
    float low_w1 = (1 - low_x.distance) * (1 - low_y.distance),
          low_w2 = (1 - low_x.distance) * low_y.distance,
          low_w3 = low_x.distance * (1 - low_y.distance),
          low_w4 = 1 - low_w1 - low_w2 - low_w3,
          high_w1 = (1 - high_x.distance) * (1 - high_y.distance),
          high_w2 = (1 - high_x.distance) * high_y.distance,
          high_w3 = high_x.distance * (1 - high_y.distance),
          high_w4 = 1 - high_w1 - high_w2 - high_w3;

    atomicAdd(&dL_dmipmaps[level.low][0][low_y.low][low_x.low][c], (1. - level.distance) * low_w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.low][0][low_y.high][low_x.low][c], (1. - level.distance) * low_w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.low][0][low_y.low][low_x.high][c], (1. - level.distance) * low_w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.low][0][low_y.high][low_x.high][c], (1. - level.distance) * low_w4 * dL_dfeat_interp[n][c]);

    atomicAdd(&dL_dmipmaps[level.high][0][high_y.low][high_x.low][c], level.distance * high_w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.high][0][high_y.high][high_x.low][c], level.distance * high_w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.high][0][high_y.low][high_x.high][c], level.distance * high_w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level.high][0][high_y.high][high_x.high][c], level.distance * high_w4 * dL_dfeat_interp[n][c]);

    // feat_interp[n][c] = (1. - level.distance) * (low_w1 * mipmaps[level.low][0][low_y.low][low_x.low][c] +
    //                                              low_w2 * mipmaps[level.low][0][low_y.high][low_x.low][c] +
    //                                              low_w3 * mipmaps[level.low][0][low_y.low][low_x.high][c] +
    //                                              low_w4 * mipmaps[level.low][0][low_y.high][low_x.high][c]) +
    //                     level.distance * (high_w1 * mipmaps[level.high][0][high_y.low][high_x.low][c] +
    //                                       high_w2 * mipmaps[level.high][0][high_y.high][high_x.low][c] +
    //                                       high_w3 * mipmaps[level.high][0][high_y.low][high_x.high][c] +
    //                                       high_w4 * mipmaps[level.high][0][high_y.high][high_x.high][c]);

    // palne XZ
    low_w1 = (1 - low_x.distance) * (1 - low_z.distance),
    low_w2 = (1 - low_x.distance) * low_z.distance,
    low_w3 = low_x.distance * (1 - low_z.distance),
    low_w4 = 1 - low_w1 - low_w2 - low_w3,
    high_w1 = (1 - high_x.distance) * (1 - high_z.distance),
    high_w2 = (1 - high_x.distance) * high_z.distance,
    high_w3 = high_x.distance * (1 - high_z.distance),
    high_w4 = 1 - high_w1 - high_w2 - high_w3;

    atomicAdd(&dL_dmipmaps[level.low][1][low_z.low][low_x.low][c], (1. - level.distance) * low_w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.low][1][low_z.high][low_x.low][c], (1. - level.distance) * low_w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.low][1][low_z.low][low_x.high][c], (1. - level.distance) * low_w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.low][1][low_z.high][low_x.high][c], (1. - level.distance) * low_w4 * dL_dfeat_interp[n][c + channel_num]);

    atomicAdd(&dL_dmipmaps[level.high][1][high_z.low][high_x.low][c], level.distance * high_w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][1][high_z.high][high_x.low][c], level.distance * high_w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][1][high_z.low][high_x.high][c], level.distance * high_w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][1][high_z.high][high_x.high][c], level.distance * high_w4 * dL_dfeat_interp[n][c + channel_num]);

    // feat_interp[n][c + channel_num] = (1. - level.distance) * (low_w1 * mipmaps[level.low][1][low_z.low][low_x.low][c] +
    //                                                            low_w2 * mipmaps[level.low][1][low_z.high][low_x.low][c] +
    //                                                            low_w3 * mipmaps[level.low][1][low_z.low][low_x.high][c] +
    //                                                            low_w4 * mipmaps[level.low][1][low_z.high][low_x.high][c]) +
    //                                   level.distance * (high_w1 * mipmaps[level.high][1][high_z.low][high_x.low][c] +
    //                                                     high_w2 * mipmaps[level.high][1][high_z.high][high_x.low][c] +
    //                                                     high_w3 * mipmaps[level.high][1][high_z.low][high_x.high][c] +
    //                                                     high_w4 * mipmaps[level.high][1][high_z.high][high_x.high][c]);

    // palne YZ
    low_w1 = (1 - low_y_w.distance) * (1 - low_z.distance),
    low_w2 = (1 - low_y_w.distance) * low_z.distance,
    low_w3 = low_y_w.distance * (1 - low_z.distance),
    low_w4 = 1 - low_w1 - low_w2 - low_w3,
    high_w1 = (1 - high_y_w.distance) * (1 - high_z.distance),
    high_w2 = (1 - high_y_w.distance) * high_z.distance,
    high_w3 = high_y_w.distance * (1 - high_z.distance),
    high_w4 = 1 - high_w1 - high_w2 - high_w3;

    atomicAdd(&dL_dmipmaps[level.low][2][low_z.low][low_y_w.low][c], (1. - level.distance) * low_w1 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dmipmaps[level.low][2][low_z.high][low_y_w.low][c], (1. - level.distance) * low_w2 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dmipmaps[level.low][2][low_z.low][low_y_w.high][c], (1. - level.distance) * low_w3 * dL_dfeat_interp[n][c + channel_num * 2]);
    atomicAdd(&dL_dmipmaps[level.low][2][low_z.high][low_y_w.high][c], (1. - level.distance) * low_w4 * dL_dfeat_interp[n][c + channel_num * 2]);

    atomicAdd(&dL_dmipmaps[level.high][2][high_z.low][high_y_w.low][c], level.distance * high_w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][2][high_z.high][high_y_w.low][c], level.distance * high_w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][2][high_z.low][high_y_w.high][c], level.distance * high_w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level.high][2][high_z.high][high_y_w.high][c], level.distance * high_w4 * dL_dfeat_interp[n][c + channel_num]);

    // feat_interp[n][c + channel_num * 2] = (1. - level.distance) * (low_w1 * mipmaps[level.low][2][low_z.low][low_y_w.low][c] +
    //                                                                low_w2 * mipmaps[level.low][2][low_z.high][low_y_w.low][c] +
    //                                                                low_w3 * mipmaps[level.low][2][low_z.low][low_y_w.high][c] +
    //                                                                low_w4 * mipmaps[level.low][2][low_z.high][low_y_w.high][c]) +
    //                                       level.distance * (high_w1 * mipmaps[level.high][2][high_z.low][high_y_w.low][c] +
    //                                                         high_w2 * mipmaps[level.high][2][high_z.high][high_y_w.low][c] +
    //                                                         high_w3 * mipmaps[level.high][2][high_z.low][high_y_w.high][c] +
    //                                                         high_w4 * mipmaps[level.high][2][high_z.high][high_y_w.high][c]);
}

torch::Tensor trimip_fw(
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points)               // Nx4 (xyz+level)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3);

    torch::Tensor feat_interp = torch::empty({N, C * 3}, mipmaps[0].options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    // AT_DISPATCH_FLOATING_TYPES(mipmaps[0].type(), "trimip_fw",
    //                            ([&]
    //                             { trimip_fw_kernel<<<blocks, threads>>>(
    //                                   mipmaps_dev,
    //                                   (int)mipmaps.size(),
    //                                   points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
    //                                   feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()); }));
    trimip_fw_kernel<<<blocks, threads>>>(
        mipmaps_dev,
        (int)mipmaps.size(),
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());

    cudaFree(mipmaps_dev);
    return feat_interp;
}

std::vector<torch::Tensor> trimip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points)               // Nx4 (xyz+level)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3);

    std::vector<torch::Tensor> dL_dmipmaps;
    for (auto &it : mipmaps)
        dL_dmipmaps.emplace_back(torch::zeros({it.size(0), it.size(1), it.size(2), it.size(3)}, it.options()));

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev, *dL_dmipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    cudaMalloc(&dL_dmipmaps_dev, sizeof(dL_dmipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host, dL_dmipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    for (auto &it : dL_dmipmaps)
        dL_dmipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dL_dmipmaps_dev, dL_dmipmaps_host.data(), sizeof(dL_dmipmaps_host[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    trimip_bw_kernel<<<blocks, threads>>>(
        dL_dfeat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        mipmaps_dev,
        (int)mipmaps.size(),
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        dL_dmipmaps_dev);

    cudaFree(mipmaps_dev);
    cudaFree(dL_dmipmaps_dev);
    return dL_dmipmaps;
}

// template <typename scalar_t, typename scalar_t2>
__global__ void tri_anisomip_fw_kernel(
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (3xHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx6 (xyz+level_xyz)
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> feat_interp)  // Nx3C
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    // palne XY
    InterpCoord_2D level_loc = locate_pt_2d(points[n][3], points[n][4], level_num, level_num);

    int level_idx_l1 = level2idx(level_loc.p1, level_num);
    int level_idx_l2 = level2idx(level_loc.p2, level_num);
    int level_idx_l3 = level2idx(level_loc.p3, level_num);
    int level_idx_l4 = level2idx(level_loc.p4, level_num);

    int height_l1 = mipmaps[level_idx_l1].size(1),
        width_l1 = mipmaps[level_idx_l1].size(2),
        height_l2 = mipmaps[level_idx_l2].size(1),
        width_l2 = mipmaps[level_idx_l2].size(2),
        height_l3 = mipmaps[level_idx_l3].size(1),
        width_l3 = mipmaps[level_idx_l3].size(2),
        height_l4 = mipmaps[level_idx_l4].size(1),
        width_l4 = mipmaps[level_idx_l4].size(2);

    InterpCoord_2D point_loc_l1 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l1 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l1 - 0.5,
                       width_l1,
                       height_l1),
                   point_loc_l2 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l2 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l2 - 0.5,
                       width_l2,
                       height_l2),
                   point_loc_l3 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l3 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l3 - 0.5,
                       width_l3,
                       height_l3),
                   point_loc_l4 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l4 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l4 - 0.5,
                       width_l4,
                       height_l4);

    feat_interp[n][c] = level_loc.w1 * (point_loc_l1.w1 * mipmaps[level_idx_l1][0][point_loc_l1.p1.v][point_loc_l1.p1.u][c] +
                                        point_loc_l1.w2 * mipmaps[level_idx_l1][0][point_loc_l1.p2.v][point_loc_l1.p2.u][c] +
                                        point_loc_l1.w3 * mipmaps[level_idx_l1][0][point_loc_l1.p3.v][point_loc_l1.p3.u][c] +
                                        point_loc_l1.w4 * mipmaps[level_idx_l1][0][point_loc_l1.p4.v][point_loc_l1.p4.u][c]) +
                        level_loc.w2 * (point_loc_l2.w1 * mipmaps[level_idx_l2][0][point_loc_l2.p1.v][point_loc_l2.p1.u][c] +
                                        point_loc_l2.w2 * mipmaps[level_idx_l2][0][point_loc_l2.p2.v][point_loc_l2.p2.u][c] +
                                        point_loc_l2.w3 * mipmaps[level_idx_l2][0][point_loc_l2.p3.v][point_loc_l2.p3.u][c] +
                                        point_loc_l2.w4 * mipmaps[level_idx_l2][0][point_loc_l2.p4.v][point_loc_l2.p4.u][c]) +
                        level_loc.w3 * (point_loc_l3.w1 * mipmaps[level_idx_l3][0][point_loc_l3.p1.v][point_loc_l3.p1.u][c] +
                                        point_loc_l3.w2 * mipmaps[level_idx_l3][0][point_loc_l3.p2.v][point_loc_l3.p2.u][c] +
                                        point_loc_l3.w3 * mipmaps[level_idx_l3][0][point_loc_l3.p3.v][point_loc_l3.p3.u][c] +
                                        point_loc_l3.w4 * mipmaps[level_idx_l3][0][point_loc_l3.p4.v][point_loc_l3.p4.u][c]) +
                        level_loc.w4 * (point_loc_l4.w1 * mipmaps[level_idx_l4][0][point_loc_l4.p1.v][point_loc_l4.p1.u][c] +
                                        point_loc_l4.w2 * mipmaps[level_idx_l4][0][point_loc_l4.p2.v][point_loc_l4.p2.u][c] +
                                        point_loc_l4.w3 * mipmaps[level_idx_l4][0][point_loc_l4.p3.v][point_loc_l4.p3.u][c] +
                                        point_loc_l4.w4 * mipmaps[level_idx_l4][0][point_loc_l4.p4.v][point_loc_l4.p4.u][c]);

    // palne XZ
    level_loc = locate_pt_2d(points[n][3], points[n][5], level_num, level_num);

    level_idx_l1 = level2idx(level_loc.p1, level_num);
    level_idx_l2 = level2idx(level_loc.p2, level_num);
    level_idx_l3 = level2idx(level_loc.p3, level_num);
    level_idx_l4 = level2idx(level_loc.p4, level_num);

    height_l1 = mipmaps[level_idx_l1].size(1),
    width_l1 = mipmaps[level_idx_l1].size(2),
    height_l2 = mipmaps[level_idx_l2].size(1),
    width_l2 = mipmaps[level_idx_l2].size(2),
    height_l3 = mipmaps[level_idx_l3].size(1),
    width_l3 = mipmaps[level_idx_l3].size(2),
    height_l4 = mipmaps[level_idx_l4].size(1),
    width_l4 = mipmaps[level_idx_l4].size(2);

    point_loc_l1 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l1 - 0.5,
        (points[n][2] + 1.) / 2. * height_l1 - 0.5,
        width_l1,
        height_l1),
    point_loc_l2 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l2 - 0.5,
        (points[n][2] + 1.) / 2. * height_l2 - 0.5,
        width_l2,
        height_l2),
    point_loc_l3 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l3 - 0.5,
        (points[n][2] + 1.) / 2. * height_l3 - 0.5,
        width_l3,
        height_l3),
    point_loc_l4 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l4 - 0.5,
        (points[n][2] + 1.) / 2. * height_l4 - 0.5,
        width_l4,
        height_l4);

    feat_interp[n][c + channel_num] = level_loc.w1 * (point_loc_l1.w1 * mipmaps[level_idx_l1][1][point_loc_l1.p1.v][point_loc_l1.p1.u][c] +
                                                      point_loc_l1.w2 * mipmaps[level_idx_l1][1][point_loc_l1.p2.v][point_loc_l1.p2.u][c] +
                                                      point_loc_l1.w3 * mipmaps[level_idx_l1][1][point_loc_l1.p3.v][point_loc_l1.p3.u][c] +
                                                      point_loc_l1.w4 * mipmaps[level_idx_l1][1][point_loc_l1.p4.v][point_loc_l1.p4.u][c]) +
                                      level_loc.w2 * (point_loc_l2.w1 * mipmaps[level_idx_l2][1][point_loc_l2.p1.v][point_loc_l2.p1.u][c] +
                                                      point_loc_l2.w2 * mipmaps[level_idx_l2][1][point_loc_l2.p2.v][point_loc_l2.p2.u][c] +
                                                      point_loc_l2.w3 * mipmaps[level_idx_l2][1][point_loc_l2.p3.v][point_loc_l2.p3.u][c] +
                                                      point_loc_l2.w4 * mipmaps[level_idx_l2][1][point_loc_l2.p4.v][point_loc_l2.p4.u][c]) +
                                      level_loc.w3 * (point_loc_l3.w1 * mipmaps[level_idx_l3][1][point_loc_l3.p1.v][point_loc_l3.p1.u][c] +
                                                      point_loc_l3.w2 * mipmaps[level_idx_l3][1][point_loc_l3.p2.v][point_loc_l3.p2.u][c] +
                                                      point_loc_l3.w3 * mipmaps[level_idx_l3][1][point_loc_l3.p3.v][point_loc_l3.p3.u][c] +
                                                      point_loc_l3.w4 * mipmaps[level_idx_l3][1][point_loc_l3.p4.v][point_loc_l3.p4.u][c]) +
                                      level_loc.w4 * (point_loc_l4.w1 * mipmaps[level_idx_l4][1][point_loc_l4.p1.v][point_loc_l4.p1.u][c] +
                                                      point_loc_l4.w2 * mipmaps[level_idx_l4][1][point_loc_l4.p2.v][point_loc_l4.p2.u][c] +
                                                      point_loc_l4.w3 * mipmaps[level_idx_l4][1][point_loc_l4.p3.v][point_loc_l4.p3.u][c] +
                                                      point_loc_l4.w4 * mipmaps[level_idx_l4][1][point_loc_l4.p4.v][point_loc_l4.p4.u][c]);

    // palne YZ
    level_loc = locate_pt_2d(points[n][4], points[n][5], level_num, level_num);

    level_idx_l1 = level2idx(level_loc.p1, level_num);
    level_idx_l2 = level2idx(level_loc.p2, level_num);
    level_idx_l3 = level2idx(level_loc.p3, level_num);
    level_idx_l4 = level2idx(level_loc.p4, level_num);

    height_l1 = mipmaps[level_idx_l1].size(1),
    width_l1 = mipmaps[level_idx_l1].size(2),
    height_l2 = mipmaps[level_idx_l2].size(1),
    width_l2 = mipmaps[level_idx_l2].size(2),
    height_l3 = mipmaps[level_idx_l3].size(1),
    width_l3 = mipmaps[level_idx_l3].size(2),
    height_l4 = mipmaps[level_idx_l4].size(1),
    width_l4 = mipmaps[level_idx_l4].size(2);

    point_loc_l1 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l1 - 0.5,
        (points[n][2] + 1.) / 2. * height_l1 - 0.5,
        width_l1,
        height_l1),
    point_loc_l2 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l2 - 0.5,
        (points[n][2] + 1.) / 2. * height_l2 - 0.5,
        width_l2,
        height_l2),
    point_loc_l3 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l3 - 0.5,
        (points[n][2] + 1.) / 2. * height_l3 - 0.5,
        width_l3,
        height_l3),
    point_loc_l4 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l4 - 0.5,
        (points[n][2] + 1.) / 2. * height_l4 - 0.5,
        width_l4,
        height_l4);

    feat_interp[n][c + channel_num * 2] = level_loc.w1 * (point_loc_l1.w1 * mipmaps[level_idx_l1][2][point_loc_l1.p1.v][point_loc_l1.p1.u][c] +
                                                          point_loc_l1.w2 * mipmaps[level_idx_l1][2][point_loc_l1.p2.v][point_loc_l1.p2.u][c] +
                                                          point_loc_l1.w3 * mipmaps[level_idx_l1][2][point_loc_l1.p3.v][point_loc_l1.p3.u][c] +
                                                          point_loc_l1.w4 * mipmaps[level_idx_l1][2][point_loc_l1.p4.v][point_loc_l1.p4.u][c]) +
                                          level_loc.w2 * (point_loc_l2.w1 * mipmaps[level_idx_l2][2][point_loc_l2.p1.v][point_loc_l2.p1.u][c] +
                                                          point_loc_l2.w2 * mipmaps[level_idx_l2][2][point_loc_l2.p2.v][point_loc_l2.p2.u][c] +
                                                          point_loc_l2.w3 * mipmaps[level_idx_l2][2][point_loc_l2.p3.v][point_loc_l2.p3.u][c] +
                                                          point_loc_l2.w4 * mipmaps[level_idx_l2][2][point_loc_l2.p4.v][point_loc_l2.p4.u][c]) +
                                          level_loc.w3 * (point_loc_l3.w1 * mipmaps[level_idx_l3][2][point_loc_l3.p1.v][point_loc_l3.p1.u][c] +
                                                          point_loc_l3.w2 * mipmaps[level_idx_l3][2][point_loc_l3.p2.v][point_loc_l3.p2.u][c] +
                                                          point_loc_l3.w3 * mipmaps[level_idx_l3][2][point_loc_l3.p3.v][point_loc_l3.p3.u][c] +
                                                          point_loc_l3.w4 * mipmaps[level_idx_l3][2][point_loc_l3.p4.v][point_loc_l3.p4.u][c]) +
                                          level_loc.w4 * (point_loc_l4.w1 * mipmaps[level_idx_l4][2][point_loc_l4.p1.v][point_loc_l4.p1.u][c] +
                                                          point_loc_l4.w2 * mipmaps[level_idx_l4][2][point_loc_l4.p2.v][point_loc_l4.p2.u][c] +
                                                          point_loc_l4.w3 * mipmaps[level_idx_l4][2][point_loc_l4.p3.v][point_loc_l4.p3.u][c] +
                                                          point_loc_l4.w4 * mipmaps[level_idx_l4][2][point_loc_l4.p4.v][point_loc_l4.p4.u][c]);
}


// template <typename scalar_t>
__global__ void tri_anisomip_bw_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (3xHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx6 (xyz+level)
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *dL_dmipmaps)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3);

    if (n >= points.size(0) || c >= channel_num)
        return;

    // palne XY
    InterpCoord_2D level_loc = locate_pt_2d(points[n][3], points[n][4], level_num, level_num);

    int level_idx_l1 = level2idx(level_loc.p1, level_num);
    int level_idx_l2 = level2idx(level_loc.p2, level_num);
    int level_idx_l3 = level2idx(level_loc.p3, level_num);
    int level_idx_l4 = level2idx(level_loc.p4, level_num);

    int height_l1 = mipmaps[level_idx_l1].size(1),
        width_l1 = mipmaps[level_idx_l1].size(2),
        height_l2 = mipmaps[level_idx_l2].size(1),
        width_l2 = mipmaps[level_idx_l2].size(2),
        height_l3 = mipmaps[level_idx_l3].size(1),
        width_l3 = mipmaps[level_idx_l3].size(2),
        height_l4 = mipmaps[level_idx_l4].size(1),
        width_l4 = mipmaps[level_idx_l4].size(2);

    InterpCoord_2D point_loc_l1 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l1 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l1 - 0.5,
                       width_l1,
                       height_l1),
                   point_loc_l2 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l2 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l2 - 0.5,
                       width_l2,
                       height_l2),
                   point_loc_l3 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l3 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l3 - 0.5,
                       width_l3,
                       height_l3),
                   point_loc_l4 = locate_pt_2d(
                       (points[n][0] + 1.) / 2. * width_l4 - 0.5,
                       (points[n][1] + 1.) / 2. * height_l4 - 0.5,
                       width_l4,
                       height_l4);

    atomicAdd(&dL_dmipmaps[level_idx_l1][0][point_loc_l1.p1.v][point_loc_l1.p1.u][c], level_loc.w1 * point_loc_l1.w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][0][point_loc_l1.p2.v][point_loc_l1.p2.u][c], level_loc.w1 * point_loc_l1.w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][0][point_loc_l1.p3.v][point_loc_l1.p3.u][c], level_loc.w1 * point_loc_l1.w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][0][point_loc_l1.p4.v][point_loc_l1.p4.u][c], level_loc.w1 * point_loc_l1.w4 * dL_dfeat_interp[n][c]);

    atomicAdd(&dL_dmipmaps[level_idx_l2][0][point_loc_l2.p1.v][point_loc_l2.p1.u][c], level_loc.w2 * point_loc_l2.w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][0][point_loc_l2.p2.v][point_loc_l2.p2.u][c], level_loc.w2 * point_loc_l2.w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][0][point_loc_l2.p3.v][point_loc_l2.p3.u][c], level_loc.w2 * point_loc_l2.w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][0][point_loc_l2.p4.v][point_loc_l2.p4.u][c], level_loc.w2 * point_loc_l2.w4 * dL_dfeat_interp[n][c]);

    atomicAdd(&dL_dmipmaps[level_idx_l3][0][point_loc_l3.p1.v][point_loc_l3.p1.u][c], level_loc.w3 * point_loc_l3.w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][0][point_loc_l3.p2.v][point_loc_l3.p2.u][c], level_loc.w3 * point_loc_l3.w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][0][point_loc_l3.p3.v][point_loc_l3.p3.u][c], level_loc.w3 * point_loc_l3.w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][0][point_loc_l3.p4.v][point_loc_l3.p4.u][c], level_loc.w3 * point_loc_l3.w4 * dL_dfeat_interp[n][c]);

    atomicAdd(&dL_dmipmaps[level_idx_l4][0][point_loc_l4.p1.v][point_loc_l4.p1.u][c], level_loc.w4 * point_loc_l4.w1 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][0][point_loc_l4.p2.v][point_loc_l4.p2.u][c], level_loc.w4 * point_loc_l4.w2 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][0][point_loc_l4.p3.v][point_loc_l4.p3.u][c], level_loc.w4 * point_loc_l4.w3 * dL_dfeat_interp[n][c]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][0][point_loc_l4.p4.v][point_loc_l4.p4.u][c], level_loc.w4 * point_loc_l4.w4 * dL_dfeat_interp[n][c]);
   
    // palne XZ
    level_loc = locate_pt_2d(points[n][3], points[n][5], level_num, level_num);

    level_idx_l1 = level2idx(level_loc.p1, level_num);
    level_idx_l2 = level2idx(level_loc.p2, level_num);
    level_idx_l3 = level2idx(level_loc.p3, level_num);
    level_idx_l4 = level2idx(level_loc.p4, level_num);

    height_l1 = mipmaps[level_idx_l1].size(1),
    width_l1 = mipmaps[level_idx_l1].size(2),
    height_l2 = mipmaps[level_idx_l2].size(1),
    width_l2 = mipmaps[level_idx_l2].size(2),
    height_l3 = mipmaps[level_idx_l3].size(1),
    width_l3 = mipmaps[level_idx_l3].size(2),
    height_l4 = mipmaps[level_idx_l4].size(1),
    width_l4 = mipmaps[level_idx_l4].size(2);

    point_loc_l1 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l1 - 0.5,
        (points[n][2] + 1.) / 2. * height_l1 - 0.5,
        width_l1,
        height_l1),
    point_loc_l2 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l2 - 0.5,
        (points[n][2] + 1.) / 2. * height_l2 - 0.5,
        width_l2,
        height_l2),
    point_loc_l3 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l3 - 0.5,
        (points[n][2] + 1.) / 2. * height_l3 - 0.5,
        width_l3,
        height_l3),
    point_loc_l4 = locate_pt_2d(
        (points[n][0] + 1.) / 2. * width_l4 - 0.5,
        (points[n][2] + 1.) / 2. * height_l4 - 0.5,
        width_l4,
        height_l4);

    atomicAdd(&dL_dmipmaps[level_idx_l1][1][point_loc_l1.p1.v][point_loc_l1.p1.u][c], level_loc.w1 * point_loc_l1.w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][1][point_loc_l1.p2.v][point_loc_l1.p2.u][c], level_loc.w1 * point_loc_l1.w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][1][point_loc_l1.p3.v][point_loc_l1.p3.u][c], level_loc.w1 * point_loc_l1.w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][1][point_loc_l1.p4.v][point_loc_l1.p4.u][c], level_loc.w1 * point_loc_l1.w4 * dL_dfeat_interp[n][c + channel_num]);

    atomicAdd(&dL_dmipmaps[level_idx_l2][1][point_loc_l2.p1.v][point_loc_l2.p1.u][c], level_loc.w2 * point_loc_l2.w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][1][point_loc_l2.p2.v][point_loc_l2.p2.u][c], level_loc.w2 * point_loc_l2.w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][1][point_loc_l2.p3.v][point_loc_l2.p3.u][c], level_loc.w2 * point_loc_l2.w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][1][point_loc_l2.p4.v][point_loc_l2.p4.u][c], level_loc.w2 * point_loc_l2.w4 * dL_dfeat_interp[n][c + channel_num]);

    atomicAdd(&dL_dmipmaps[level_idx_l3][1][point_loc_l3.p1.v][point_loc_l3.p1.u][c], level_loc.w3 * point_loc_l3.w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][1][point_loc_l3.p2.v][point_loc_l3.p2.u][c], level_loc.w3 * point_loc_l3.w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][1][point_loc_l3.p3.v][point_loc_l3.p3.u][c], level_loc.w3 * point_loc_l3.w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][1][point_loc_l3.p4.v][point_loc_l3.p4.u][c], level_loc.w3 * point_loc_l3.w4 * dL_dfeat_interp[n][c + channel_num]);

    atomicAdd(&dL_dmipmaps[level_idx_l4][1][point_loc_l4.p1.v][point_loc_l4.p1.u][c], level_loc.w4 * point_loc_l4.w1 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][1][point_loc_l4.p2.v][point_loc_l4.p2.u][c], level_loc.w4 * point_loc_l4.w2 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][1][point_loc_l4.p3.v][point_loc_l4.p3.u][c], level_loc.w4 * point_loc_l4.w3 * dL_dfeat_interp[n][c + channel_num]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][1][point_loc_l4.p4.v][point_loc_l4.p4.u][c], level_loc.w4 * point_loc_l4.w4 * dL_dfeat_interp[n][c + channel_num]);

    // palne YZ
    level_loc = locate_pt_2d(points[n][4], points[n][5], level_num, level_num);

    level_idx_l1 = level2idx(level_loc.p1, level_num);
    level_idx_l2 = level2idx(level_loc.p2, level_num);
    level_idx_l3 = level2idx(level_loc.p3, level_num);
    level_idx_l4 = level2idx(level_loc.p4, level_num);

    height_l1 = mipmaps[level_idx_l1].size(1),
    width_l1 = mipmaps[level_idx_l1].size(2),
    height_l2 = mipmaps[level_idx_l2].size(1),
    width_l2 = mipmaps[level_idx_l2].size(2),
    height_l3 = mipmaps[level_idx_l3].size(1),
    width_l3 = mipmaps[level_idx_l3].size(2),
    height_l4 = mipmaps[level_idx_l4].size(1),
    width_l4 = mipmaps[level_idx_l4].size(2);

    point_loc_l1 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l1 - 0.5,
        (points[n][2] + 1.) / 2. * height_l1 - 0.5,
        width_l1,
        height_l1),
    point_loc_l2 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l2 - 0.5,
        (points[n][2] + 1.) / 2. * height_l2 - 0.5,
        width_l2,
        height_l2),
    point_loc_l3 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l3 - 0.5,
        (points[n][2] + 1.) / 2. * height_l3 - 0.5,
        width_l3,
        height_l3),
    point_loc_l4 = locate_pt_2d(
        (points[n][1] + 1.) / 2. * width_l4 - 0.5,
        (points[n][2] + 1.) / 2. * height_l4 - 0.5,
        width_l4,
        height_l4);

    atomicAdd(&dL_dmipmaps[level_idx_l1][2][point_loc_l1.p1.v][point_loc_l1.p1.u][c], level_loc.w1 * point_loc_l1.w1 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][2][point_loc_l1.p2.v][point_loc_l1.p2.u][c], level_loc.w1 * point_loc_l1.w2 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][2][point_loc_l1.p3.v][point_loc_l1.p3.u][c], level_loc.w1 * point_loc_l1.w3 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l1][2][point_loc_l1.p4.v][point_loc_l1.p4.u][c], level_loc.w1 * point_loc_l1.w4 * dL_dfeat_interp[n][c + channel_num*2]);

    atomicAdd(&dL_dmipmaps[level_idx_l2][2][point_loc_l2.p1.v][point_loc_l2.p1.u][c], level_loc.w2 * point_loc_l2.w1 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][2][point_loc_l2.p2.v][point_loc_l2.p2.u][c], level_loc.w2 * point_loc_l2.w2 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][2][point_loc_l2.p3.v][point_loc_l2.p3.u][c], level_loc.w2 * point_loc_l2.w3 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l2][2][point_loc_l2.p4.v][point_loc_l2.p4.u][c], level_loc.w2 * point_loc_l2.w4 * dL_dfeat_interp[n][c + channel_num*2]);

    atomicAdd(&dL_dmipmaps[level_idx_l3][2][point_loc_l3.p1.v][point_loc_l3.p1.u][c], level_loc.w3 * point_loc_l3.w1 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][2][point_loc_l3.p2.v][point_loc_l3.p2.u][c], level_loc.w3 * point_loc_l3.w2 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][2][point_loc_l3.p3.v][point_loc_l3.p3.u][c], level_loc.w3 * point_loc_l3.w3 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l3][2][point_loc_l3.p4.v][point_loc_l3.p4.u][c], level_loc.w3 * point_loc_l3.w4 * dL_dfeat_interp[n][c + channel_num*2]);

    atomicAdd(&dL_dmipmaps[level_idx_l4][2][point_loc_l4.p1.v][point_loc_l4.p1.u][c], level_loc.w4 * point_loc_l4.w1 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][2][point_loc_l4.p2.v][point_loc_l4.p2.u][c], level_loc.w4 * point_loc_l4.w2 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][2][point_loc_l4.p3.v][point_loc_l4.p3.u][c], level_loc.w4 * point_loc_l4.w3 * dL_dfeat_interp[n][c + channel_num*2]);
    atomicAdd(&dL_dmipmaps[level_idx_l4][2][point_loc_l4.p4.v][point_loc_l4.p4.u][c], level_loc.w4 * point_loc_l4.w4 * dL_dfeat_interp[n][c + channel_num*2]);
}



torch::Tensor tri_anisomip_fw(
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points)               // Nx6 (xyz+level_xyz)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3);

    torch::Tensor feat_interp = torch::empty({N, C * 3}, mipmaps[0].options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    int num_levels = int(sqrt(mipmaps.size()));

    // AT_DISPATCH_FLOATING_TYPES(mipmaps[0].type(), "trimip_fw",
    //                            ([&]
    //                             { trimip_fw_kernel<<<blocks, threads>>>(
    //                                   mipmaps_dev,
    //                                   (int)mipmaps.size(),
    //                                   points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
    //                                   feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()); }));
    tri_anisomip_fw_kernel<<<blocks, threads>>>(
        mipmaps_dev,
        num_levels,
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());

    cudaFree(mipmaps_dev);
    return feat_interp;
}


std::vector<torch::Tensor> tri_anisomip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points)               // Nx6 (xyz+level)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3);

    std::vector<torch::Tensor> dL_dmipmaps;
    for (auto &it : mipmaps)
        dL_dmipmaps.emplace_back(torch::zeros({it.size(0), it.size(1), it.size(2), it.size(3)}, it.options()));

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev, *dL_dmipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    cudaMalloc(&dL_dmipmaps_dev, sizeof(dL_dmipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host, dL_dmipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    for (auto &it : dL_dmipmaps)
        dL_dmipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dL_dmipmaps_dev, dL_dmipmaps_host.data(), sizeof(dL_dmipmaps_host[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    int num_levels = int(sqrt(mipmaps.size()));

    tri_anisomip_bw_kernel<<<blocks, threads>>>(
        dL_dfeat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        mipmaps_dev,
        num_levels,
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        dL_dmipmaps_dev);

    cudaFree(mipmaps_dev);
    cudaFree(dL_dmipmaps_dev);
    return dL_dmipmaps;
}



// 
// template <typename scalar_t, typename scalar_t2>
__global__ void multi_anisomip_fw_kernel(
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (MxHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx4M (xy+level_xy)
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> feat_interp)  // NxMC
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3),
              plane_num = mipmaps[0].size(0);

    if (n >= points.size(0) || c >= channel_num)
        return;
    
    for (auto plane_id=0; plane_id<plane_num; plane_id++)
    {
        InterpCoord_2D level_loc = locate_pt_2d(points[n][4*plane_id + 2], points[n][4*plane_id + 3], level_num, level_num);
        int level_idx_l1 = level2idx(level_loc.p1, level_num);
        int level_idx_l2 = level2idx(level_loc.p2, level_num);
        int level_idx_l3 = level2idx(level_loc.p3, level_num);
        int level_idx_l4 = level2idx(level_loc.p4, level_num);

        int height_l1 = mipmaps[level_idx_l1].size(1),
            width_l1 = mipmaps[level_idx_l1].size(2),
            height_l2 = mipmaps[level_idx_l2].size(1),
            width_l2 = mipmaps[level_idx_l2].size(2),
            height_l3 = mipmaps[level_idx_l3].size(1),
            width_l3 = mipmaps[level_idx_l3].size(2),
            height_l4 = mipmaps[level_idx_l4].size(1),
            width_l4 = mipmaps[level_idx_l4].size(2);

        InterpCoord_2D point_loc_l1 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l1 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l1 - 0.5,
                        width_l1,
                        height_l1),
                    point_loc_l2 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l2 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l2 - 0.5,
                        width_l2,
                        height_l2),
                    point_loc_l3 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l3 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l3 - 0.5,
                        width_l3,
                        height_l3),
                    point_loc_l4 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l4 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l4 - 0.5,
                        width_l4,
                        height_l4);

        feat_interp[n][c + plane_id * channel_num] = level_loc.w1 * (point_loc_l1.w1 * mipmaps[level_idx_l1][plane_id][point_loc_l1.p1.v][point_loc_l1.p1.u][c] +
                                            point_loc_l1.w2 * mipmaps[level_idx_l1][plane_id][point_loc_l1.p2.v][point_loc_l1.p2.u][c] +
                                            point_loc_l1.w3 * mipmaps[level_idx_l1][plane_id][point_loc_l1.p3.v][point_loc_l1.p3.u][c] +
                                            point_loc_l1.w4 * mipmaps[level_idx_l1][plane_id][point_loc_l1.p4.v][point_loc_l1.p4.u][c]) +
                            level_loc.w2 * (point_loc_l2.w1 * mipmaps[level_idx_l2][plane_id][point_loc_l2.p1.v][point_loc_l2.p1.u][c] +
                                            point_loc_l2.w2 * mipmaps[level_idx_l2][plane_id][point_loc_l2.p2.v][point_loc_l2.p2.u][c] +
                                            point_loc_l2.w3 * mipmaps[level_idx_l2][plane_id][point_loc_l2.p3.v][point_loc_l2.p3.u][c] +
                                            point_loc_l2.w4 * mipmaps[level_idx_l2][plane_id][point_loc_l2.p4.v][point_loc_l2.p4.u][c]) +
                            level_loc.w3 * (point_loc_l3.w1 * mipmaps[level_idx_l3][plane_id][point_loc_l3.p1.v][point_loc_l3.p1.u][c] +
                                            point_loc_l3.w2 * mipmaps[level_idx_l3][plane_id][point_loc_l3.p2.v][point_loc_l3.p2.u][c] +
                                            point_loc_l3.w3 * mipmaps[level_idx_l3][plane_id][point_loc_l3.p3.v][point_loc_l3.p3.u][c] +
                                            point_loc_l3.w4 * mipmaps[level_idx_l3][plane_id][point_loc_l3.p4.v][point_loc_l3.p4.u][c]) +
                            level_loc.w4 * (point_loc_l4.w1 * mipmaps[level_idx_l4][plane_id][point_loc_l4.p1.v][point_loc_l4.p1.u][c] +
                                            point_loc_l4.w2 * mipmaps[level_idx_l4][plane_id][point_loc_l4.p2.v][point_loc_l4.p2.u][c] +
                                            point_loc_l4.w3 * mipmaps[level_idx_l4][plane_id][point_loc_l4.p3.v][point_loc_l4.p3.u][c] +
                                            point_loc_l4.w4 * mipmaps[level_idx_l4][plane_id][point_loc_l4.p4.v][point_loc_l4.p4.u][c]);
    }
    
}


// template <typename scalar_t>
__global__ void multi_anisomip_bw_kernel(
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps, // list of tensor (MxHxWxC)
    int mipmap_levels,
    const torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> points, // Nx4M (xy+level_xy)
    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *dL_dmipmaps)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;

    const int level_num = mipmap_levels,
              channel_num = mipmaps[0].size(3),
                plane_num = mipmaps[0].size(0);


    if (n >= points.size(0) || c >= channel_num)
        return;
    
    for (auto plane_id=0; plane_id<plane_num; plane_id++)
    {
        InterpCoord_2D level_loc = locate_pt_2d(points[n][4*plane_id + 2], points[n][4*plane_id + 3], level_num, level_num);
        int level_idx_l1 = level2idx(level_loc.p1, level_num);
        int level_idx_l2 = level2idx(level_loc.p2, level_num);
        int level_idx_l3 = level2idx(level_loc.p3, level_num);
        int level_idx_l4 = level2idx(level_loc.p4, level_num);

        int height_l1 = mipmaps[level_idx_l1].size(1),
            width_l1 = mipmaps[level_idx_l1].size(2),
            height_l2 = mipmaps[level_idx_l2].size(1),
            width_l2 = mipmaps[level_idx_l2].size(2),
            height_l3 = mipmaps[level_idx_l3].size(1),
            width_l3 = mipmaps[level_idx_l3].size(2),
            height_l4 = mipmaps[level_idx_l4].size(1),
            width_l4 = mipmaps[level_idx_l4].size(2);

        InterpCoord_2D point_loc_l1 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l1 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l1 - 0.5,
                        width_l1,
                        height_l1),
                    point_loc_l2 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l2 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l2 - 0.5,
                        width_l2,
                        height_l2),
                    point_loc_l3 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l3 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l3 - 0.5,
                        width_l3,
                        height_l3),
                    point_loc_l4 = locate_pt_2d(
                        (points[n][4*plane_id + 0] + 1.) / 2. * width_l4 - 0.5,
                        (points[n][4*plane_id + 1] + 1.) / 2. * height_l4 - 0.5,
                        width_l4,
                        height_l4);
        
        atomicAdd(&dL_dmipmaps[level_idx_l1][plane_id][point_loc_l1.p1.v][point_loc_l1.p1.u][c], level_loc.w1 * point_loc_l1.w1 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l1][plane_id][point_loc_l1.p2.v][point_loc_l1.p2.u][c], level_loc.w1 * point_loc_l1.w2 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l1][plane_id][point_loc_l1.p3.v][point_loc_l1.p3.u][c], level_loc.w1 * point_loc_l1.w3 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l1][plane_id][point_loc_l1.p4.v][point_loc_l1.p4.u][c], level_loc.w1 * point_loc_l1.w4 * dL_dfeat_interp[n][c + plane_id * channel_num]);

        atomicAdd(&dL_dmipmaps[level_idx_l2][plane_id][point_loc_l2.p1.v][point_loc_l2.p1.u][c], level_loc.w2 * point_loc_l2.w1 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l2][plane_id][point_loc_l2.p2.v][point_loc_l2.p2.u][c], level_loc.w2 * point_loc_l2.w2 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l2][plane_id][point_loc_l2.p3.v][point_loc_l2.p3.u][c], level_loc.w2 * point_loc_l2.w3 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l2][plane_id][point_loc_l2.p4.v][point_loc_l2.p4.u][c], level_loc.w2 * point_loc_l2.w4 * dL_dfeat_interp[n][c + plane_id * channel_num]);

        atomicAdd(&dL_dmipmaps[level_idx_l3][plane_id][point_loc_l3.p1.v][point_loc_l3.p1.u][c], level_loc.w3 * point_loc_l3.w1 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l3][plane_id][point_loc_l3.p2.v][point_loc_l3.p2.u][c], level_loc.w3 * point_loc_l3.w2 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l3][plane_id][point_loc_l3.p3.v][point_loc_l3.p3.u][c], level_loc.w3 * point_loc_l3.w3 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l3][plane_id][point_loc_l3.p4.v][point_loc_l3.p4.u][c], level_loc.w3 * point_loc_l3.w4 * dL_dfeat_interp[n][c + plane_id * channel_num]);

        atomicAdd(&dL_dmipmaps[level_idx_l4][plane_id][point_loc_l4.p1.v][point_loc_l4.p1.u][c], level_loc.w4 * point_loc_l4.w1 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l4][plane_id][point_loc_l4.p2.v][point_loc_l4.p2.u][c], level_loc.w4 * point_loc_l4.w2 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l4][plane_id][point_loc_l4.p3.v][point_loc_l4.p3.u][c], level_loc.w4 * point_loc_l4.w3 * dL_dfeat_interp[n][c + plane_id * channel_num]);
        atomicAdd(&dL_dmipmaps[level_idx_l4][plane_id][point_loc_l4.p4.v][point_loc_l4.p4.u][c], level_loc.w4 * point_loc_l4.w4 * dL_dfeat_interp[n][c + plane_id * channel_num]);
    }

}



torch::Tensor multi_anisomip_fw(
    const std::vector<torch::Tensor> mipmaps, // list of tensor (MxHxWxC)
    const torch::Tensor points)               // Nx4M (xy+level_xy)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3), M = mipmaps[0].size(0);

    torch::Tensor feat_interp = torch::empty({N, C * M}, mipmaps[0].options());

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    int num_levels = int(sqrt(mipmaps.size()));

    // AT_DISPATCH_FLOATING_TYPES(mipmaps[0].type(), "trimip_fw",
    //                            ([&]
    //                             { trimip_fw_kernel<<<blocks, threads>>>(
    //                                   mipmaps_dev,
    //                                   (int)mipmaps.size(),
    //                                   points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
    //                                   feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>()); }));
    multi_anisomip_fw_kernel<<<blocks, threads>>>(
        mipmaps_dev,
        num_levels,
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>());

    cudaFree(mipmaps_dev);
    return feat_interp;
}

std::vector<torch::Tensor> multi_anisomip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (MxHxWxC)
    const torch::Tensor points)               // Nx4M (xy+level_xy)
{
    for (int i = 0; i < mipmaps.size(); i++)
    {
        CHECK_INPUT(mipmaps[i]);
    }
    CHECK_INPUT(points);

    const int N = points.size(0), C = mipmaps[0].size(3), M = mipmaps[0].size(0);

    std::vector<torch::Tensor> dL_dmipmaps;
    for (auto &it : mipmaps)
        dL_dmipmaps.emplace_back(torch::zeros({it.size(0), it.size(1), it.size(2), it.size(3)}, it.options()));

    const dim3 threads(16, THREADS);
    const dim3 blocks((N + threads.x - 1) / threads.x, (C + threads.y - 1) / threads.y);

    // printf("%d", sizeof(mipmaps[0].type()));

    torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t> *mipmaps_dev, *dL_dmipmaps_dev;
    cudaMalloc(&mipmaps_dev, sizeof(mipmaps_dev[0]) * mipmaps.size());
    cudaMalloc(&dL_dmipmaps_dev, sizeof(dL_dmipmaps_dev[0]) * mipmaps.size());
    std::vector<torch::PackedTensorAccessor<float, 4, torch::RestrictPtrTraits, size_t>> mipmaps_host, dL_dmipmaps_host;
    for (auto &it : mipmaps)
        mipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    for (auto &it : dL_dmipmaps)
        dL_dmipmaps_host.emplace_back(it.packed_accessor<float, 4, torch::RestrictPtrTraits, size_t>());
    cudaMemcpy(mipmaps_dev, mipmaps_host.data(), sizeof(mipmaps_dev[0]) * mipmaps.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dL_dmipmaps_dev, dL_dmipmaps_host.data(), sizeof(dL_dmipmaps_host[0]) * mipmaps.size(), cudaMemcpyHostToDevice);

    int num_levels = int(sqrt(mipmaps.size()));

    multi_anisomip_bw_kernel<<<blocks, threads>>>(
        dL_dfeat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        mipmaps_dev,
        num_levels,
        points.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        dL_dmipmaps_dev);

    cudaFree(mipmaps_dev);
    cudaFree(dL_dmipmaps_dev);
    return dL_dmipmaps;
}
