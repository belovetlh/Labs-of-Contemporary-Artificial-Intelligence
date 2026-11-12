"""Grad-CAM (Selvaraju et al., 2017)：用目标类别得分对卷积特征图的梯度定位模型关注的区域。"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """对指定卷积层计算类激活热力图。

    步骤：
    1. 前向传播，记录目标层输出的特征图 A（形状 [N, K, h, w]）；
    2. 对目标类别的 logit 反向传播，得到 dY_c / dA；
    3. 对梯度做全局平均，得到每个通道的权重 alpha_k；
    4. CAM = ReLU(sum_k alpha_k * A_k)，再上采样到输入尺寸并归一化到 [0, 1]。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor = None) -> np.ndarray:
        """返回形状 [N, H, W] 的热力图。class_idx 为 None 时解释模型预测的类别。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        # eval 模式下各样本互不影响，把 batch 内的目标得分求和后一次反传即可
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        return cam.squeeze(1).detach().cpu().numpy()

    def remove(self) -> None:
        """移除钩子，避免影响模型后续的正常使用。"""
        self._handle.remove()
