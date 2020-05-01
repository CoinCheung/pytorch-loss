
#include <torch/extension.h>

// definitions
at::Tensor Mish_forward_cuda(const at::Tensor &feat);
at::Tensor Mish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat);

// python inferface
at::Tensor Mish_forward(const at::Tensor &feat) {
    if (!feat.type().is_cuda()) {
        AT_ERROR("this mish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Mish_forward_cuda(feat);
}

at::Tensor Mish_backward(const at::Tensor &grad, const at::Tensor &feat) {
    // TODO: try AT_ASSERTM
    if (!feat.type().is_cuda()) {
        AT_ERROR("this mish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Mish_backward_cuda(grad, feat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mish_forward", &Mish_forward, "mish forward");
    m.def("mish_backward", &Mish_backward, "mish backward");
}
