"""可解释性分析入口：Grad-CAM 热力图与中间层特征图可视化。

示例：
    python explain.py --run outputs/base_resnet18

在 <run>/explain/ 下生成：
    gradcam_correct.png  每个类别一张分类正确的样本及其 Grad-CAM
    gradcam_errors.png   置信度最高的误分类样本：分别对"预测类别"和"真实类别"画 Grad-CAM
    feature_maps.png     一张图像在网络各阶段的特征图（仅从零训练的 CNN）
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CLASSES, build_dataloaders
from src.engine import evaluate, load_trained_model
from src.gradcam import GradCAM
from src.models import is_pretrained
from src.utils import get_device


def overlay(ax, image: np.ndarray, cam: np.ndarray, title: str) -> None:
    """把热力图半透明叠加在原图上。"""
    ax.imshow(image)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot_gradcam_correct(cam: GradCAM, inputs, images, targets, preds, path: Path) -> None:
    """每个类别取第一张分类正确的测试样本，上排原图、下排 Grad-CAM。"""
    picks = [int(np.where((targets == c) & (preds == c))[0][0])
             for c in range(len(CLASSES)) if ((targets == c) & (preds == c)).any()]
    cams = cam(inputs(picks))
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 1.7, 3.8), squeeze=False)
    for col, i in enumerate(picks):
        show(axes[0, col], images[i], CLASSES[targets[i]])
        overlay(axes[1, col], images[i], cams[col], "Grad-CAM")
    fig.suptitle("Grad-CAM on correctly classified samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam_errors(cam: GradCAM, inputs, images, targets, probs, path: Path, n: int = 8) -> None:
    """对置信度最高的 n 个误分类样本，对比模型"看预测类别"与"看真实类别"时关注的区域。"""
    preds, confs = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.where(preds != targets)[0]
    wrong = wrong[np.argsort(-confs[wrong])][:n].tolist()
    if not wrong:
        return
    x = inputs(wrong)
    cams_pred = cam(x, torch.as_tensor(preds[wrong], device=x.device))
    cams_true = cam(x, torch.as_tensor(targets[wrong], device=x.device))
    fig, axes = plt.subplots(3, len(wrong), figsize=(len(wrong) * 1.9, 6), squeeze=False)
    for col, i in enumerate(wrong):
        show(axes[0, col], images[i], f"true: {CLASSES[targets[i]]}")
        overlay(axes[1, col], images[i], cams_pred[col],
                f"pred: {CLASSES[preds[i]]} ({confs[i]:.2f})")
        overlay(axes[2, col], images[i], cams_true[col], f"CAM for {CLASSES[targets[i]]}")
    fig.suptitle("Grad-CAM on misclassified samples (row 2: predicted class, row 3: true class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_feature_maps(model, x: torch.Tensor, image: np.ndarray, path: Path, channels: int = 8) -> None:
    """展示一张图像在各阶段输出的前若干个通道，直观观察特征由浅层纹理到深层语义的变化。"""
    layers = {name: m for name, m in model.named_children()
              if name.startswith(("stem", "stage", "layer"))}
    if not layers:
        print("该模型没有可直接可视化的卷积阶段（预训练封装模型），跳过特征图。")
        return
    feats = {}
    hooks = [m.register_forward_hook(lambda _m, _i, out, name=name: feats.__setitem__(name, out))
             for name, m in layers.items()]
    model(x)
    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(len(feats), channels + 1, squeeze=False,
                             figsize=((channels + 1) * 1.5, len(feats) * 1.6))
    for row, (name, feat) in enumerate(feats.items()):
        show(axes[row, 0], image, f"{name}\n{tuple(feat.shape[1:])}")
        for col in range(channels):
            axes[row, col + 1].imshow(feat[0, col].float().cpu().numpy(), cmap="viridis")
            axes[row, col + 1].axis("off")
    fig.suptitle("Feature maps at each stage (first channels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM 与特征图可视化")
    p.add_argument("--run", required=True, type=Path, help="训练输出目录")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    device = get_device()
    model, config = load_trained_model(args.run, device)
    if model.cam_layer is None:
        raise SystemExit(f"{config['model']} 没有卷积特征图，Grad-CAM 只支持 CNN 模型。")
    norm = "imagenet" if is_pretrained(config["model"]) else "cifar"
    _, _, test_loader, images = build_dataloaders(
        args.data_dir, "none", norm, 256, args.workers, debug=config["debug"])
    _, _, probs, targets = evaluate(model, test_loader, device)
    probs, targets = probs.numpy(), targets.numpy()
    preds = probs.argmax(axis=1)

    def inputs(indices):
        """取出指定测试样本的归一化张量并组成 batch。"""
        return torch.stack([test_loader.dataset[i][0] for i in indices]).to(device)

    out_dir = args.run / "explain"
    out_dir.mkdir(exist_ok=True)
    cam = GradCAM(model, model.cam_layer)
    plot_gradcam_correct(cam, inputs, images, targets, preds, out_dir / "gradcam_correct.png")
    plot_gradcam_errors(cam, inputs, images, targets, probs, out_dir / "gradcam_errors.png")
    cam.remove()
    plot_feature_maps(model, inputs([0]), images[0], out_dir / "feature_maps.png")
    print(f"可解释性图表已保存到 {out_dir}")


if __name__ == "__main__":
    main()
