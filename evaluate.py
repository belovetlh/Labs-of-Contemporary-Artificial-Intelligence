"""评估入口：加载已训练好的模型，在测试集上重新评估并输出全部指标与图表。

示例：
    python evaluate.py --run outputs/base_resnet18
    python evaluate.py --run outputs/base_resnet18 --tta     # 使用测试时增强

结果保存在 <run>/eval/（或 eval_tta/）下，不会覆盖训练时生成的文件。
"""
import argparse
from pathlib import Path

from src.data import CLASSES, build_dataloaders
from src.engine import evaluate, load_trained_model
from src.models import is_pretrained
from src.utils import get_device, save_json
from src.visualize import test_report


def main() -> None:
    p = argparse.ArgumentParser(description="在 CIFAR-10 测试集上评估已训练的模型")
    p.add_argument("--run", required=True, type=Path, help="训练输出目录，如 outputs/base_resnet18")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tta", action="store_true", help="对原图与水平翻转图的预测取平均")
    args = p.parse_args()

    device = get_device()
    model, config = load_trained_model(args.run, device)
    norm = "imagenet" if is_pretrained(config["model"]) else "cifar"
    # 这里只用到测试集；debug 标志沿用训练时的设置，保证样本数一致
    _, _, test_loader, test_images = build_dataloaders(
        args.data_dir, "none", norm, args.batch_size, args.workers, debug=config["debug"])

    test_loss, _, probs, targets = evaluate(model, test_loader, device,
                                            use_amp=device.type == "cuda", tta=args.tta)
    out_dir = args.run / ("eval_tta" if args.tta else "eval")
    out_dir.mkdir(exist_ok=True)
    report = test_report(probs.numpy(), targets.numpy(), test_images, CLASSES, out_dir)
    save_json({"test_loss": test_loss, "tta": args.tta, **report}, out_dir / "results.json")

    print(f"模型 {config['model']} | 测试样本 {len(targets)} | TTA {args.tta}")
    print(f"总准确率: {report['test_acc'] * 100:.2f}%   测试损失: {test_loss:.4f}")
    print("每类准确率:")
    for cls, acc in report["per_class_acc"].items():
        print(f"  {cls:<11s} {acc * 100:.2f}%")
    print("最易混淆的类别对 (真实 -> 预测: 数量):")
    for c in report["top_confusions"]:
        print(f"  {c['true']} -> {c['pred']}: {c['count']}")
    print(f"图表已保存到 {out_dir}")


if __name__ == "__main__":
    main()
