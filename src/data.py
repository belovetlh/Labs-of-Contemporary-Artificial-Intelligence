"""数据管线：CIFAR-10 加载、训练/验证集分层划分、数据增强、Mixup / CutMix。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import seed_worker

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# 归一化统计量：从零训练的模型用 CIFAR-10 自身的均值/方差，
# 预训练模型沿用 ImageNet 的统计量，与其预训练时的输入分布保持一致。
NORM_STATS = {
    "cifar": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

AUG_POLICIES = ("none", "basic", "cutout", "autoaugment", "randaugment")


class Cutout:
    """Cutout (DeVries & Taylor, 2017)：随机遮挡一个正方形区域，迫使模型利用全局信息。

    作用在已归一化的张量上，遮挡值 0 即对应数据集均值。
    """

    def __init__(self, length: int = 16):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
        x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_transforms(aug: str, norm: str):
    """根据增强策略名称构建训练/评估两套变换。评估阶段只做归一化。

    none         不做增强
    basic        随机裁剪(padding=4) + 随机水平翻转
    cutout       basic + Cutout(16)
    autoaugment  basic + AutoAugment(CIFAR-10 策略)
    randaugment  basic + RandAugment(N=2, M=9)
    """
    if aug not in AUG_POLICIES:
        raise ValueError(f"未知的增强策略: {aug}，可选 {AUG_POLICIES}")
    mean, std = NORM_STATS[norm]
    to_tensor = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_ops = []
    if aug != "none":
        train_ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if aug == "autoaugment":
        train_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    elif aug == "randaugment":
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_ops += to_tensor
    if aug == "cutout":
        train_ops.append(Cutout(16))

    return transforms.Compose(train_ops), transforms.Compose(to_tensor)


def stratified_split(targets, val_size: int, seed: int):
    """按类别分层划分训练/验证索引，保证验证集中各类样本数相同。

    只依赖 NumPy 的 RandomState，因此相同 seed 在任何机器上得到完全相同的划分。
    """
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    classes = np.unique(targets)
    per_class = val_size // len(classes)
    train_idx, val_idx = [], []
    for c in classes:
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[:per_class].tolist())
        train_idx.extend(idx[per_class:].tolist())
    return sorted(train_idx), sorted(val_idx)


def build_dataloaders(data_dir: str, aug: str, norm: str, batch_size: int, workers: int,
                      val_size: int = 5000, split_seed: int = 42, seed: int = 42,
                      debug: bool = False, pin_memory: bool = False):
    """构建训练/验证/测试 DataLoader。

    官方 50,000 张训练图按 split_seed 分层划分为训练集与验证集；官方 10,000 张测试集
    全程不参与训练和调参，只用于最终评估。
    split_seed 只决定数据划分，seed 决定训练时的样本顺序；两者分开，
    这样换训练种子做重复实验时，所有实验仍然使用同一份验证集。

    返回 (train_loader, val_loader, test_loader, test_images)，其中 test_images 是与
    test_loader 顺序一致的原始 uint8 图像，供误分类样本与 Grad-CAM 可视化使用。
    debug=True 时只取极少量样本，用于快速检查代码能否跑通。
    """
    train_tf, eval_tf = build_transforms(aug, norm)
    # 同一份训练数据建两个视图：训练子集带增强，验证子集不带增强
    train_view = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_view = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = stratified_split(train_view.targets, val_size, split_seed)
    test_idx = list(range(len(test_set)))
    if debug:
        train_idx, val_idx, test_idx = train_idx[:512], val_idx[:256], test_idx[:256]

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory,
                  worker_init_fn=seed_worker, persistent_workers=workers > 0)
    train_loader = DataLoader(Subset(train_view, train_idx), shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(Subset(val_view, val_idx), shuffle=False, **common)
    test_loader = DataLoader(Subset(test_set, test_idx), shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_set.data[test_idx]


def _rand_bbox(h: int, w: int, lam: float):
    """CutMix 用：按面积比例 (1 - lam) 随机生成一个裁剪框。"""
    ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * ratio), int(w * ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    return y1, y2, x1, x2


def mix_batch(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float, cutmix_alpha: float):
    """对一个 batch 做 Mixup 或 CutMix（两者都启用时各以 50% 概率选择）。

    返回 (混合后的图像, 标签 a, 标签 b, lam)，损失为
    lam * L(pred, a) + (1 - lam) * L(pred, b)。
    """
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    perm = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        # 裁剪框可能被图像边界截断，按实际面积修正 lam
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1.0 - lam) * x[perm]
    return x, y, y[perm], lam
