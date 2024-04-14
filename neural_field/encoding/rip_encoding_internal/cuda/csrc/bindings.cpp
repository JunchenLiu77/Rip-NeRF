#include <torch/extension.h>

torch::Tensor bilinear_fw(
    const torch::Tensor grid_feats,
    const torch::Tensor points);

torch::Tensor bilinear_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor grid_feats,
    const torch::Tensor points);

torch::Tensor triplane_fw(
    const torch::Tensor feats,
    const torch::Tensor points);

torch::Tensor triplane_bw(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points);

torch::Tensor trimip_fw(
    const std::vector<torch::Tensor> mipmap,
    const torch::Tensor points);

std::vector<torch::Tensor> trimip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points);               // Nx4 (xyz+level)

torch::Tensor tri_anisomip_fw(
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points);               // Nx6 (xyz+level)

std::vector<torch::Tensor> tri_anisomip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (3xHxWxC)
    const torch::Tensor points);               // Nx6 (xyz+level)

torch::Tensor multi_anisomip_fw(
    const std::vector<torch::Tensor> mipmaps, // list of tensor (MxHxWxC)
    const torch::Tensor points);               // Nx4M (xy+level_xy)

std::vector<torch::Tensor> multi_anisomip_bw(
    const torch::Tensor dL_dfeat_interp,
    const std::vector<torch::Tensor> mipmaps, // list of tensor (MxHxWxC)
    const torch::Tensor points);               // Nx4M (xy+level_xy)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("bilinear_fw", &bilinear_fw);
    m.def("bilinear_bw", &bilinear_bw);
    m.def("triplane_fw", &triplane_fw);
    m.def("triplane_bw", &triplane_bw);
    m.def("trimip_fw", &trimip_fw);
    m.def("trimip_bw", &trimip_bw);
    m.def("tri_anisomip_fw", &tri_anisomip_fw);
    m.def("tri_anisomip_bw", &tri_anisomip_bw);
    m.def("multi_anisomip_fw", &multi_anisomip_fw);
    m.def("multi_anisomip_bw", &multi_anisomip_bw);
}