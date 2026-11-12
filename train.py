"""训练入口：训练一个模型，用验证集选出最优权重，最后在测试集上评估一次。

示例：
    python train.py --name baseline --model simple_cnn --dropout 0.3
    python train.py --name quick_check --debug --epochs 1 --workers 0   # 快速检查代码能否跑通

所有产物保存在 outputs/<name>/ 下：
    config.json / train.log / history.json / curves.png
    best.pt（验证集最优权重）/ last.pt（断点续训用）
    results.json / confusion_matrix.png / per_class_acc.png / misclassified.png
"""
import argparse
import time
from pathlib import Path

import torch

from src.data import AUG_POLICIES, CLASSES, build_dataloaders
from src.engine import (SCHEDULERS, EarlyStopping, build_optimizer, build_scheduler,
                        evaluate, train_one_epoch)
from src.losses import build_criterion
from src.models import MODEL_NAMES, build_model, is_pretrained
from src.utils import (count_parameters, env_info, get_device, save_json, set_seed,
                       setup_logger)
from src.visualize import plot_curves, test_report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 图像分类训练脚本",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", required=True, help="实验名称，输出保存到 <output-dir>/<name>/")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--data-dir", default="data", help="CIFAR-10 存放目录，不存在时自动下载")

    g = p.add_argument_group("模型")
    g.add_argument("--model", default="resnet18", choices=MODEL_NAMES)
    g.add_argument("--dropout", type=float, default=0.0, help="分类头前的 Dropout 概率")
    g.add_argument("--no-pretrained-weights", action="store_true",
                   help="预训练模型不下载 ImageNet 权重（仅用于离线检查代码）")

    g = p.add_argument_group("数据与增强")
    g.add_argument("--aug", default="basic", choices=AUG_POLICIES)
    g.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup 的 Beta 分布参数，0 表示不用")
    g.add_argument("--cutmix-alpha", type=float, default=0.0, help="CutMix 的 Beta 分布参数，0 表示不用")
    g.add_argument("--val-size", type=int, default=5000, help="从 50,000 张训练图中划出的验证集大小")
    g.add_argument("--split-seed", type=int, default=42,
                   help="训练/验证划分的随机种子，与 --seed 分开，保证所有实验共用同一份验证集")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--workers", type=int, default=4, help="DataLoader 子进程数")

    g = p.add_argument_group("损失函数")
    g.add_argument("--loss", default="ce", choices=("ce", "focal"))
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--focal-gamma", type=float, default=2.0)

    g = p.add_argument_group("优化")
    g.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw"))
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--scheduler", default="cosine", choices=SCHEDULERS)
    g.add_argument("--warmup-epochs", type=int, default=0, help="cosine 调度前的线性 warmup 轮数")
    g.add_argument("--early-stop-patience", type=int, default=0,
                   help="验证准确率连续多少轮不提升就停止，0 表示不启用早停")

    g = p.add_argument_group("运行")
    g.add_argument("--seed", type=int, default=42, help="权重初始化、样本顺序、数据增强的随机种子")
    g.add_argument("--no-amp", action="store_true", help="关闭混合精度（默认在 GPU 上启用）")
    g.add_argument("--no-deterministic", action="store_true",
                   help="关闭 cuDNN 确定性模式以换取速度")
    g.add_argument("--tta", action="store_true", help="最终测试时使用水平翻转的测试时增强")
    g.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    g.add_argument("--debug", action="store_true", help="只用极少量数据，快速检查代码能否跑通")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.output_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, last_path, best_path = (out_dir / "results.json", out_dir / "last.pt",
                                          out_dir / "best.pt")
    logger = setup_logger(out_dir / "train.log")
    if args.resume and results_path.exists():
        logger.info(f"实验 {args.name} 已完成，跳过。")
        return

    set_seed(args.seed, deterministic=not args.no_deterministic)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    save_json(vars(args), out_dir / "config.json")
    logger.info(f"实验 {args.name} | 设备 {device} | 混合精度 {use_amp}")
    logger.info(f"配置: {vars(args)}")
    if args.debug:
        logger.info("注意：debug 模式只使用极少量数据，结果没有参考意义。")

    # ---------- 数据 ----------
    norm = "imagenet" if is_pretrained(args.model) else "cifar"
    train_loader, val_loader, test_loader, test_images = build_dataloaders(
        args.data_dir, args.aug, norm, args.batch_size, args.workers,
        val_size=args.val_size, split_seed=args.split_seed, seed=args.seed, debug=args.debug,
        pin_memory=device.type == "cuda")
    logger.info(f"训练集 {len(train_loader.dataset)} | 验证集 {len(val_loader.dataset)} | "
                f"测试集 {len(test_loader.dataset)}")

    # ---------- 模型、损失、优化器、调度器 ----------
    model = build_model(args.model, dropout=args.dropout,
                        load_weights=not args.no_pretrained_weights)
    model = model.to(device, memory_format=torch.channels_last)
    num_params = count_parameters(model)
    logger.info(f"模型 {args.model} | 参数量 {num_params / 1e6:.2f}M")

    criterion = build_criterion(args.loss, args.label_smoothing, args.focal_gamma)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler, step_per_batch = build_scheduler(
        optimizer, args.scheduler, args.epochs, len(train_loader), args.lr, args.warmup_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(args.early_stop_patience)

    # ---------- 断点续训 ----------
    history, start_epoch, best_acc, best_epoch = [], 0, -1.0, 0
    if args.resume and last_path.exists():
        # last.pt 是本脚本自己保存的文件，包含优化器/调度器状态，需要完整反序列化
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        stopper.best, stopper.counter = ckpt["stopper"]
        history, best_acc, best_epoch = ckpt["history"], ckpt["best_acc"], ckpt["best_epoch"]
        start_epoch = len(history)
        logger.info(f"从第 {start_epoch} 个 epoch 之后继续训练")

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if stopper.should_stop:
            break
        tic = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            args.mixup_alpha, args.cutmix_alpha,
            batch_scheduler=scheduler if step_per_batch else None)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, use_amp)

        # 按 epoch 更新的调度器：plateau 需要验证指标，其余直接 step
        if scheduler is not None and not step_per_batch:
            if args.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "time": time.time() - tic})
        logger.info(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | "
                    f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
                    f"val loss {val_loss:.4f} acc {val_acc * 100:.2f}% | {time.time() - tic:.0f}s")

        # 只用验证集准确率挑选最优权重，测试集不参与任何选择
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)
        stop = stopper.step(val_acc)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "stopper": (stopper.best, stopper.counter), "history": history,
                    "best_acc": best_acc, "best_epoch": best_epoch}, last_path)
        save_json(history, out_dir / "history.json")
        if stop:
            logger.info(f"早停：验证准确率已连续 {stopper.patience} 轮未提升，在第 {epoch} 轮停止。")

    # ---------- 用验证集最优权重在测试集上做最终评估 ----------
    plot_curves(history, out_dir / "curves.png")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])
    test_loss, _, probs, targets = evaluate(model, test_loader, device, use_amp, tta=args.tta)
    report = test_report(probs.numpy(), targets.numpy(), test_images, CLASSES, out_dir)

    results = {
        "name": args.name, "model": args.model, "params": num_params,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_acc": best_acc,
        "test_loss": test_loss, **report,
        "train_minutes": sum(h["time"] for h in history) / 60,
        "config": vars(args), "env": env_info(),
    }
    save_json(results, results_path)
    logger.info(f"最优验证准确率 {best_acc * 100:.2f}%（第 {best_epoch} 轮）| "
                f"测试准确率 {report['test_acc'] * 100:.2f}%")
    for cls, acc in report["per_class_acc"].items():
        logger.info(f"  {cls:<11s} {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
