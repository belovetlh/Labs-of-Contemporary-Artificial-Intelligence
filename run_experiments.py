"""批量运行对比实验。每组实验只改变一个因素，其余设置与基线 base_resnet18 保持一致。

示例：
    python run_experiments.py --group arch          # 运行"架构对比"一组
    python run_experiments.py --group hparam        # 基线 CNN 的超参数搜索
    python run_experiments.py --group all           # 依次运行全部实验
    python run_experiments.py --group aug --dry-run # 只打印命令，不执行

已完成的实验会自动跳过，被中断的实验会从断点继续，因此可以放心地分多次运行。
"""
import argparse
import subprocess
import sys

# 基线：train.py 的默认参数
#   ResNet-18(CIFAR 版) / basic 增强 / SGD lr=0.1 wd=5e-4 / cosine 调度 / 交叉熵 / 60 epoch / batch 128
BASE = {"base_resnet18": {}}

# 自行设计的基线 CNN，使用默认超参数（lr=0.1, wd=5e-4）
SIMPLE_CNN = {"arch_simple_cnn": {"model": "simple_cnn", "dropout": 0.3}}

GROUPS = {
    # 1. 架构对比：自行设计的基线 CNN、ResNet-18，以及加入 SE / CBAM 注意力的变体
    "arch": {
        **SIMPLE_CNN,
        **BASE,
        "arch_resnet18_se": {"model": "resnet18_se"},
        "arch_resnet18_cbam": {"model": "resnet18_cbam"},
    },
    # 超参数搜索：在基线 CNN 上对学习率 x 权重衰减做网格搜索，用验证集准确率挑选。
    # 其中 lr=0.1, wd=5e-4 这一格就是 arch_simple_cnn，不重复训练。
    "hparam": {
        **SIMPLE_CNN,
        **{f"hp_lr{lr}_wd{wd}": {"model": "simple_cnn", "dropout": 0.3, "lr": lr, "weight_decay": wd}
           for lr in (0.01, 0.05, 0.1) for wd in (1e-4, 5e-4) if (lr, wd) != (0.1, 5e-4)},
    },
    # 随机性的大小：基线只换训练种子重复 3 次（数据划分不变），
    # 得到的标准差用来判断其他对比实验中的差距是否超出了随机波动。
    "seeds": {
        **BASE,  # seed=42
        "base_resnet18_seed1": {"seed": 1},
        "base_resnet18_seed2": {"seed": 2},
    },
    # 2. ImageNet 预训练模型微调（输入上采样到 224x224，学习率更小、轮数更少）
    "pretrained": {
        "pt_resnet18": {"model": "resnet18_pt", "lr": 0.01, "epochs": 15, "warmup_epochs": 1},
        "pt_mobilenet_v2": {"model": "mobilenet_v2_pt", "lr": 0.01, "epochs": 15, "warmup_epochs": 1},
        "pt_vit_b_16": {"model": "vit_b_16_pt", "optimizer": "adamw", "lr": 5e-5,
                        "weight_decay": 0.05, "epochs": 5, "warmup_epochs": 1, "batch_size": 64},
    },
    # 3. 学习率调度与早停
    "sched": {
        "sched_none": {"scheduler": "none"},
        "sched_step": {"scheduler": "step"},
        **BASE,  # cosine
        "sched_warmup_cosine": {"scheduler": "cosine", "warmup_epochs": 5},
        "sched_onecycle": {"scheduler": "onecycle"},
        "sched_plateau_earlystop": {"scheduler": "plateau", "early_stop_patience": 12},
    },
    # 4. 数据增强策略
    "aug": {
        "aug_none": {"aug": "none"},
        **BASE,  # basic：随机裁剪 + 水平翻转
        "aug_cutout": {"aug": "cutout"},
        "aug_autoaugment": {"aug": "autoaugment"},
        "aug_mixup": {"mixup_alpha": 1.0},
        "aug_cutmix": {"cutmix_alpha": 1.0},
    },
    # 5. 损失函数
    "loss": {
        **BASE,  # 交叉熵
        "loss_label_smoothing": {"label_smoothing": 0.1},
        "loss_focal": {"loss": "focal", "focal_gamma": 2.0},
    },
    # 6. 最终模型：组合各组对比实验中在验证集上表现最好的设置，并延长训练。
    #    下面是暂定配置，应在前面各组实验跑完后根据验证集结果调整。
    "final": {
        "final_model": {"model": "resnet18_se", "aug": "autoaugment", "cutmix_alpha": 1.0,
                        "label_smoothing": 0.1, "warmup_epochs": 5, "epochs": 200, "tta": True},
    },
}


def to_argv(name: str, overrides: dict) -> list:
    """把一个实验的参数字典转换成 train.py 的命令行参数。"""
    argv = ["--name", name, "--resume"]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return argv


def main() -> None:
    p = argparse.ArgumentParser(description="批量运行对比实验")
    p.add_argument("--group", required=True, choices=[*GROUPS, "all"])
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="在 -- 之后追加给每个实验的 train.py 参数，如 -- --workers 2 --debug --epochs 1")
    args = p.parse_args()
    extra = [a for a in args.extra if a != "--"]

    groups = GROUPS.values() if args.group == "all" else [GROUPS[args.group]]
    experiments = {name: cfg for group in groups for name, cfg in group.items()}
    for i, (name, overrides) in enumerate(experiments.items(), 1):
        cmd = [sys.executable, "train.py", *to_argv(name, overrides), *extra]
        print(f"\n===== [{i}/{len(experiments)}] {' '.join(cmd)} =====", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
