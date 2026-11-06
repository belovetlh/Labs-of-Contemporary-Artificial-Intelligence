"""各实验脚本共用的输出工具。"""
from __future__ import annotations

import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mean(xs) -> float:
    return statistics.fmean(xs)


# |t| 超过约 2 时，差异由随机波动造成的可能性小于约 5%
T_THRESHOLD = 2.0


def paired_test(a: list, b: list) -> dict:
    """配对比较：a、b 是同一批样本（如同一批地图）在两种设置下的结果。

    逐个样本计算差值 d = b - a，再看差值的平均值是否明显大于它自身的波动：
      标准误 se = 差值的标准差 / sqrt(n)，表示"换一批样本，平均差大约会抖动多少"；
      t = 平均差 / se，|t| >= 2 视为显著（不太可能是碰巧）。
    """
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    diff = statistics.fmean(d)
    se = statistics.stdev(d) / n ** 0.5 if n > 1 else 0.0
    if se == 0:
        t = 0.0 if diff == 0 else float("inf") * (1 if diff > 0 else -1)
    else:
        t = diff / se
    return {"n": n, "diff": diff, "se": se, "t": t, "significant": abs(t) >= T_THRESHOLD}


def verdict(test: dict) -> str:
    return "显著" if test["significant"] else "不显著"


class Output:
    """results/<name>/ 目录下的实验产物：CSV 原始数据、PNG 图、summary.md 汇总。"""

    def __init__(self, name: str):
        self.name = name
        self.dir = os.path.join(ROOT, "results", name)

    def prepare(self, quick: bool = False) -> None:
        """quick=True（缩小规模的检查运行）时输出到 results/quick/，不覆盖正式结果。"""
        if quick:
            self.dir = os.path.join(ROOT, "results", "quick", self.name)
        os.makedirs(self.dir, exist_ok=True)

    def path(self, fname: str) -> str:
        return os.path.join(self.dir, fname)

    def write_csv(self, fname: str, header: list[str], rows: list[list]) -> None:
        # utf-8-sig 让 Excel 直接打开也不会中文乱码
        with open(self.path(fname), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def write_summary(self, sections: list[str]) -> str:
        p = self.path("summary.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n\n".join(sections) + "\n")
        return os.path.relpath(p, ROOT)
