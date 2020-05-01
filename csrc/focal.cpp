
#include <torch/extension.h>

// definition in the other file
at::Tensor FocalLoss_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha);

at::Tensor FocalLoss_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha);

// python inferface
at::Tensor FocalLoss_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    if (!logits.type().is_cuda()) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_forward_cuda(logits, labels, gamma, alpha);
}

at::Tensor FocalLoss_backward(const at::Tensor &grad,
                             const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    // TODO: try AT_ASSERTM
    if (!logits.type().is_cuda()) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_backward_cuda(grad, logits, labels, gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("focalloss_forward", &FocalLoss_forward, "focal loss forward");
    m.def("focalloss_backward", &FocalLoss_backward, "focal loss backward");
}
