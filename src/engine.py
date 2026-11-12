"""训练与评估的核心逻辑：单轮训练、评估、早停、优化器与学习率调度器的构建。"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import mix_batch
from .models import build_model
from .utils import load_json

SCHEDULERS = ("none", "step", "cosine", "onecycle", "plateau")


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """把图像搬到目标设备；channels_last 内存布局在 GPU 混合精度下卷积更快。"""
    return x.to(device, non_blocking=True, memory_format=torch.channels_last)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0, batch_scheduler=None):
    """训练一个 epoch，返回 (平均损失, 准确率)。

    scaler           混合精度的梯度缩放器；未启用时等价于普通的 backward + step
    batch_scheduler  需要每个 batch 更新一次的学习率调度器（cosine / onecycle），否则为 None

    使用 Mixup / CutMix 时标签是两个类别的混合，训练准确率按 lam 加权统计。
    """
    model.train()
    use_mix = mixup_alpha > 0 or cutmix_alpha > 0
    loss_sum, correct_sum, count = 0.0, 0.0, 0
    for x, y in loader:
        x, y = _to_device(x, device), y.to(device, non_blocking=True)
        y_a, y_b, lam = y, y, 1.0
        if use_mix:
            x, y_a, y_b, lam = mix_batch(x, y, mixup_alpha, cutmix_alpha)

        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            logits = model(x)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_scheduler is not None:
            batch_scheduler.step()

        pred = logits.argmax(dim=1)
        correct = lam * (pred == y_a).sum().item() + (1.0 - lam) * (pred == y_b).sum().item()
        loss_sum += loss.item() * y.size(0)
        correct_sum += correct
        count += y.size(0)
    return loss_sum / count, correct_sum / count


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = False, tta: bool = False):
    """在给定数据上评估，返回 (损失, 准确率, 概率矩阵[N, C], 标签[N])。

    评估损失统一使用不带标签平滑的交叉熵，使不同训练损失的实验之间可以直接比较。
    tta=True 时启用测试时增强：对原图与水平翻转图的预测概率取平均。
    """
    model.eval()
    probs_list, targets_list = [], []
    for x, y in loader:
        x = _to_device(x, device)
        with torch.autocast(device.type, enabled=use_amp):
            probs = model(x).float().softmax(dim=1)
            if tta:
                probs = (probs + model(torch.flip(x, dims=[3])).float().softmax(dim=1)) / 2
        probs_list.append(probs.cpu())
        targets_list.append(y)
    probs, targets = torch.cat(probs_list), torch.cat(targets_list)
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), targets).item()
    acc = (probs.argmax(dim=1) == targets).float().mean().item()
    return loss, acc, probs, targets


class EarlyStopping:
    """早停：验证准确率连续 patience 个 epoch 没有提升就停止训练。patience <= 0 表示不启用。"""

    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.counter = 0

    @property
    def should_stop(self) -> bool:
        return 0 < self.patience <= self.counter

    def step(self, metric: float) -> bool:
        """传入本轮验证指标，返回是否应该停止。"""
        if metric > self.best + self.min_delta:
            self.best, self.counter = metric, 0
        else:
            self.counter += 1
        return self.should_stop


def build_optimizer(model, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    """sgd：带 Nesterov 动量的 SGD（从零训练 CNN 的常用选择）；adamw：适合微调 ViT。"""
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知的优化器: {name}")


def build_scheduler(optimizer, name: str, epochs: int, steps_per_epoch: int,
                    lr: float, warmup_epochs: int = 0):
    """创建学习率调度器，返回 (scheduler, 是否每个 batch 更新一次)。

    none      恒定学习率
    step      在 50% 和 75% 的 epoch 处各衰减为 0.1 倍
    cosine    余弦退火到 0，可选线性 warmup（warmup_epochs > 0）
    onecycle  OneCycle：先升到最大学习率再余弦下降
    plateau   验证准确率 5 个 epoch 不提升则学习率乘 0.1（通常配合早停使用）
    """
    if name == "none":
        return None, False
    if name == "step":
        milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1), False
    if name == "cosine":
        total, warmup = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

        def lr_factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor), True
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch), True
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5), False
    raise ValueError(f"未知的调度器: {name}，可选 {SCHEDULERS}")


def load_trained_model(run_dir: Path, device: torch.device):
    """从一次实验的输出目录重建模型并加载验证集上最优的权重，返回 (model, config)。"""
    config = load_json(run_dir / "config.json")
    model = build_model(config["model"], dropout=config["dropout"], load_weights=False)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config
