"""可视化与测试集分析：训练曲线、混淆矩阵、每类准确率、误分类样本。

图中文字统一使用英文，避免云端环境缺少中文字体导致乱码。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境（云端/后台运行）下也能保存图片
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history: list, path: Path) -> None:
    """绘制训练/验证的 loss、accuracy 曲线以及学习率变化。history 为每个 epoch 一条记录。"""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(epochs, [h["train_acc"] * 100 for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] * 100 for h in history], label="val")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy (%)")
    axes[2].plot(epochs, [h["lr"] for h in history], color="tab:green", label="lr")
    axes[2].set(title="Learning rate", xlabel="epoch", ylabel="lr")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """混淆矩阵：cm[i, j] 表示真实类别为 i、被预测为 j 的样本数。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def plot_confusion_matrix(cm: np.ndarray, classes, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks, xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="True", title="Confusion matrix (test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(per_class_acc: np.ndarray, classes, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(classes, per_class_acc * 100, color="tab:blue")
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.axhline(per_class_acc.mean() * 100, color="tab:red", linestyle="--", label="mean")
    ax.set(ylabel="accuracy (%)", title="Per-class accuracy (test set)",
           ylim=(max(0.0, per_class_acc.min() * 100 - 10), 101))
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(images: np.ndarray, targets: np.ndarray, probs: np.ndarray,
                       classes, path: Path, n: int = 24) -> None:
    """展示置信度最高的 n 个误分类样本，标题格式为 "真实类别 -> 预测类别 (置信度)"。

    高置信度的错误最值得分析：模型不是"拿不准"，而是"自信地错了"。
    """
    preds, confs = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.where(preds != targets)[0]
    wrong = wrong[np.argsort(-confs[wrong])][:n]
    if len(wrong) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(images[i])
        ax.set_title(f"{classes[targets[i]]}\n-> {classes[preds[i]]} ({confs[i]:.2f})", fontsize=8)
    fig.suptitle("Most confident misclassifications (true -> predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_report(probs: np.ndarray, targets: np.ndarray, images: np.ndarray,
                classes, out_dir: Path) -> dict:
    """汇总测试集结果：计算总准确率、每类准确率、混淆矩阵和最易混淆的类别对，并保存全部图表。

    同时把预测概率存成 test_preds.npz，后续分析无需重新推理。
    """
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(targets, preds, len(classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # 最易混淆的类别对：去掉对角线后取数值最大的 5 个格子
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    top = np.argsort(off_diag, axis=None)[::-1][:5]
    confusions = [{"true": classes[i], "pred": classes[j], "count": int(off_diag[i, j])}
                  for i, j in zip(*np.unravel_index(top, cm.shape))]

    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")
    plot_per_class_accuracy(per_class_acc, classes, out_dir / "per_class_acc.png")
    plot_misclassified(images, targets, probs, classes, out_dir / "misclassified.png")
    np.savez_compressed(out_dir / "test_preds.npz", probs=probs, targets=targets)

    return {
        "test_acc": float((preds == targets).mean()),
        "per_class_acc": {c: float(a) for c, a in zip(classes, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "top_confusions": confusions,
    }
