#include <torch/extension.h>

#define THREADS 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

struct InterpCoord
{
    int low;
    int high;
    float distance;
};

struct int_uv
{
    int u;
    int v;
};

struct InterpCoord_2D
{
    int_uv p1;
    int_uv p2;
    int_uv p3;
    int_uv p4;
    float w1;
    float w2;
    float w3;
    float w4;
};

template <typename scalar_t>
__device__ InterpCoord quantize_coord(
    const scalar_t position,
    const int total_size)
{
    // position [-1,1]
    InterpCoord interp_coord = {};

    scalar_t x = (position + 1.) / 2. * total_size - 0.5;
    const int x_floor = floor(x);

    interp_coord.low = max(int(0), x_floor);
    interp_coord.high = min(int(total_size - 1), x_floor + 1);
    interp_coord.distance = x - x_floor;
    return interp_coord;
}

__device__ InterpCoord quantize(
    const float x,
    const int total_size)
{
    // position [-1,1]
    InterpCoord interp_coord = {};

    const int x_floor = floor(x);

    interp_coord.low = max(int(0), x_floor);
    interp_coord.high = min(int(total_size - 1), x_floor + 1);
    interp_coord.distance = x - x_floor;
    return interp_coord;
}

__device__ InterpCoord_2D locate_pt_2d(
    const float x,
    const float y,
    const int width,
    const int height)
{
    InterpCoord_2D coord_2d = {};
    InterpCoord x_q = quantize(x, width),
                y_q = quantize(y, height);
    coord_2d.p1.u = x_q.low;
    coord_2d.p1.v = y_q.low;

    coord_2d.p2.u = x_q.low;
    coord_2d.p2.v = y_q.high;

    coord_2d.p3.u = x_q.high;
    coord_2d.p3.v = y_q.low;

    coord_2d.p4.u = x_q.high;
    coord_2d.p4.v = y_q.high;

    coord_2d.w1 = (1 - x_q.distance) * (1 - y_q.distance);
    coord_2d.w2 = (1 - x_q.distance) * y_q.distance;
    coord_2d.w3 = x_q.distance * (1 - y_q.distance);
    coord_2d.w4 = 1 - coord_2d.w1 - coord_2d.w2 - coord_2d.w3;

    return coord_2d;
}

__device__ int level2idx(
    int_uv level_uv,
    const int level_num)
{
    int idx = level_uv.u * level_num + level_uv.v;
    return idx;
}
