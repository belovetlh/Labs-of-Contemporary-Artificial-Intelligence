"""汇总对比实验结果：生成 Markdown 表格和验证准确率曲线对比图。

示例：
    python compare.py --group aug      # 汇总"数据增强"一组
    python compare.py --group all      # 汇总全部分组

结果保存在 outputs/_compare/<group>.md 和 <group>.png。尚未完成的实验会被跳过并提示。
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import GROUPS
from src.utils import load_json


def summarize(group: str, names: list, output_dir: Path) -> None:
    runs = {}
    for name in names:
        run_dir = output_dir / name
        if (run_dir / "results.json").exists():
            runs[name] = (load_json(run_dir / "results.json"), load_json(run_dir / "history.json"))
        else:
            print(f"[{group}] {name} 尚未完成，跳过")
    if not runs:
        return
    out_dir = output_dir / "_compare"
    out_dir.mkdir(exist_ok=True)

    # ---------- 表格 ----------
    lines = ["| experiment | model | params (M) | best val acc (%) | test acc (%) | best epoch | epochs run | train time (min) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, (res, _) in runs.items():
        lines.append(f"| {name} | {res['model']} | {res['params'] / 1e6:.2f} | "
                     f"{res['best_val_acc'] * 100:.2f} | {res['test_acc'] * 100:.2f} | "
                     f"{res['best_epoch']} | {res['epochs_run']} | {res['train_minutes']:.1f} |")
    if group == "seeds" and len(runs) > 1:
        # 重复实验：给出均值和标准差，作为判断其他对比是否显著的参照
        for key in ("best_val_acc", "test_acc"):
            accs = np.array([res[key] for res, _ in runs.values()]) * 100
            lines.append(f"\n{key}: mean {accs.mean():.2f}%, std {accs.std(ddof=1):.2f}% "
                         f"(n={len(accs)})")
    table = "\n".join(lines)
    (out_dir / f"{group}.md").write_text(table + "\n", encoding="utf-8")
    print(f"\n### {group}\n{table}")

    # ---------- 曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (_, history) in runs.items():
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["val_acc"] * 100 for h in history], label=name)
        axes[1].plot(epochs, [h["train_loss"] for h in history], label=name)
    axes[0].set(title=f"Validation accuracy ({group})", xlabel="epoch", ylabel="accuracy (%)")
    axes[1].set(title=f"Training loss ({group})", xlabel="epoch", ylabel="loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{group}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="汇总对比实验结果")
    p.add_argument("--group", required=True, choices=[*GROUPS, "all"])
    p.add_argument("--output-dir", default="outputs", type=Path)
    args = p.parse_args()
    for group in (GROUPS if args.group == "all" else [args.group]):
        summarize(group, list(GROUPS[group]), args.output_dir)


if __name__ == "__main__":
    main()
