"""题目一对照实验。

运行：python -m experiments.exp_p1           （完整实验，约 1 分钟）
      python -m experiments.exp_p1 --quick   （缩小规模，用于快速检查）
输出：results/p1/ 下的 summary.md（汇总表）、*.csv（原始数据）、*.png（图）

E1 样例：h=0 与曼哈顿的逐步过程、展开区域、节点数        —— 报告必做
E2 规模 x 障碍密度：启发函数的收益如何随地图变化
E3 平局规则：f 相同的节点先展开哪个，对节点数影响多大
E4 特殊地图：空旷地图（启发最准）与 U 形陷阱（启发被误导）
E5 加权 A* 与欧氏距离：更少的节点 vs 路径是否仍最优
E6 展开节点分解：f < C* 的节点必须展开，f = C* 的节点取决于平局规则
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

from core.grid import load_grid_problem
from experiments import viz
from experiments.analysis import classify_by_f
from experiments.common import ROOT, Output, mean, paired_test, verdict
from experiments.maze_gen import open_grid, random_grid, u_trap_grid
from problems import p1_maze

OUTPUT = Output("p1")
out = OUTPUT.path
write_csv = OUTPUT.write_csv

NAMES = {"zero": "h=0", "manhattan": "曼哈顿", "euclidean": "欧氏"}


def solve(grid, start, goal, heuristic, tie_break="low_h", weight=1.0, repeats=1, trace=False):
    """多次运行取耗时中位数；节点数等统计每次都相同，取最后一次即可。"""
    times = []
    for _ in range(repeats):
        res = p1_maze.solve(start, goal, grid, heuristic, tie_break, trace, weight)
        times.append(res.elapsed)
    res.elapsed = statistics.median(times)
    return res


def passable_cells(grid) -> int:
    return sum(t != "X" for row in grid for t in row)


# ---------------------------------------------------------------- E1 样例

def e1_sample() -> str:
    start, goal, grid = load_grid_problem(os.path.join(ROOT, "data", "p1_sample.txt"))
    runs = {h: solve(grid, start, goal, h, trace=True) for h in ("zero", "manhattan")}
    viz.compare_expansions(grid, [(NAMES[h], r) for h, r in runs.items()], out("e1_sample.png"),
                           "样例地图：数字为展开顺序，红线为最终路径", show_order=True)
    rows = [[NAMES[h], len(r.path) - 1, r.expanded, r.generated, r.max_open]
            for h, r in runs.items()]
    return "\n\n".join([
        "## E1 样例：h=0 与曼哈顿距离",
        "![](e1_sample.png)",
        viz.md_table(["启发函数", "路径步数", "展开节点", "入队次数", "开放表峰值"], rows),
        "### 曼哈顿距离逐步过程", viz.step_table_md(runs["manhattan"]),
        "### h=0 逐步过程", viz.step_table_md(runs["zero"]),
    ])


# ---------------------------------------------------------------- E2 规模 x 密度

E2_HEADERS = ["边长", "障碍密度", "可通行格", "最短路步数", "展开(h=0)", "展开(曼哈顿)", "曼哈顿/h=0",
              "h=0展开占可通行格比例", "耗时ms(h=0)", "耗时ms(曼哈顿)", "起终点曼哈顿距离"]


def _scale_experiment(sizes, densities, seeds, repeats, endpoints: str):
    """跑 规模 x 密度 x 种子 的网格实验，返回 (原始数据, 按 (边长, 密度) 汇总后的行)。"""
    raw = []
    for size in sizes:
        for dens in densities:
            for seed in seeds:
                grid, s, g = random_grid(size, dens, seed, endpoints)
                res = {h: solve(grid, s, g, h, repeats=repeats) for h in ("zero", "manhattan")}
                if len(res["zero"].path) != len(res["manhattan"].path):
                    sys.exit(f"最优性校验失败：size={size} dens={dens} seed={seed} {endpoints}")
                raw.append([size, dens, seed, passable_cells(grid), len(res["zero"].path) - 1,
                            res["zero"].expanded, res["manhattan"].expanded,
                            res["zero"].elapsed * 1000, res["manhattan"].elapsed * 1000,
                            abs(s[0] - g[0]) + abs(s[1] - g[1])])
    groups = defaultdict(list)
    for row in raw:
        groups[(row[0], row[1])].append(row)
    rows = []
    for (size, dens), rs in sorted(groups.items()):
        rows.append([size, f"{dens:.0%}", mean(r[3] for r in rs), mean(r[4] for r in rs),
                     mean(r[5] for r in rs), mean(r[6] for r in rs), mean(r[6] / r[5] for r in rs),
                     mean(r[5] / r[3] for r in rs), mean(r[7] for r in rs), mean(r[8] for r in rs),
                     mean(r[9] for r in rs)])
    return raw, rows


RAW_HEADER = ["size", "density", "seed", "passable", "path_len", "expanded_zero",
              "expanded_manhattan", "ms_zero", "ms_manhattan", "endpoint_manhattan"]


def e2_scale(sizes, densities, seeds, repeats) -> tuple[str, list]:
    raw, rows = _scale_experiment(sizes, densities, seeds, repeats, "corner")
    write_csv("e2_scale.csv", RAW_HEADER, raw)
    _plot_e2(rows, sizes, densities)
    grid, s, g = random_grid(40, 0.2, seeds[0])
    viz.compare_expansions(grid, [(NAMES[h], solve(grid, s, g, h, trace=True))
                                  for h in ("zero", "manhattan")],
                           out("e2_example.png"), "40x40、障碍 20% 的随机地图")
    return "\n\n".join([
        f"## E2 地图规模 x 障碍密度（起点左上、终点右下；每格 {len(seeds)} 张随机图取平均）",
        "![](e2_scale.png)", "![](e2_example.png)",
        viz.md_table(E2_HEADERS, rows),
    ]), rows


def e2b_random_endpoints(sizes, densities, seeds, repeats, corner_rows) -> str:
    """E2 的对照：起点和终点随机选取，检验"终点在最远角落"这一设置是否夸大了启发函数的收益。"""
    raw, rows = _scale_experiment(sizes, densities, seeds, repeats, "random")
    write_csv("e2b_random_endpoints.csv", RAW_HEADER, raw)

    corner = {(r[0], r[1]): r for r in corner_rows}
    cmp_rows = [[r[0], r[1], corner[(r[0], r[1])][7], r[7], corner[(r[0], r[1])][6], r[6]]
                for r in rows if (r[0], r[1]) in corner]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, size in enumerate(sizes):
        sel = [r for r in cmp_rows if r[0] == size]
        xs = [float(r[1].rstrip("%")) for r in sel]
        a1.plot(xs, [r[2] for r in sel], "o-", color=f"C{i}", label=f"{size}x{size} 对角")
        a1.plot(xs, [r[3] for r in sel], "s--", color=f"C{i}", label=f"{size}x{size} 随机")
        a2.plot(xs, [r[4] for r in sel], "o-", color=f"C{i}", label=f"{size}x{size} 对角")
        a2.plot(xs, [r[5] for r in sel], "s--", color=f"C{i}", label=f"{size}x{size} 随机")
    a1.set(xlabel="障碍密度 (%)", ylabel="h=0 展开的格子 / 可通行格", ylim=(0, 1.05),
           title="(a) h=0 的搜索范围")
    a2.set(xlabel="障碍密度 (%)", ylabel="曼哈顿 / h=0 展开节点比", ylim=(0, 1.05),
           title="(b) 启发函数的节省比例（越低越省）")
    a1.legend(fontsize=7, ncol=2)
    a2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out("e2b_random_endpoints.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        f"## E2b 对照：起点和终点随机选取（每格 {len(seeds)} 张随机图取平均）",
        "![](e2b_random_endpoints.png)",
        viz.md_table(E2_HEADERS, rows),
        "### 与 E2（对角起终点）的对比",
        viz.md_table(["边长", "障碍密度", "h=0展开比例(对角)", "h=0展开比例(随机)",
                      "曼哈顿/h=0(对角)", "曼哈顿/h=0(随机)"], cmp_rows),
    ])


def _plot_e2(rows, sizes, densities) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, dens in enumerate(densities):
        sel = [r for r in rows if r[1] == f"{dens:.0%}"]
        color = f"C{i}"
        a1.plot([r[0] for r in sel], [r[4] for r in sel], "o-", color=color, label=f"h=0，障碍{dens:.0%}")
        a1.plot([r[0] for r in sel], [r[5] for r in sel], "s--", color=color, label=f"曼哈顿，障碍{dens:.0%}")
    a1.set(xscale="log", yscale="log", xlabel="地图边长", ylabel="平均展开节点数",
           title="(a) 展开节点数随地图规模的变化")
    a1.legend(fontsize=7, ncol=2)
    for i, size in enumerate(sizes):
        sel = [r for r in rows if r[0] == size]
        a2.plot([float(r[1].rstrip("%")) for r in sel], [r[6] for r in sel], "o-", label=f"{size}x{size}")
    a2.set(xlabel="障碍密度 (%)", ylabel="曼哈顿 / h=0 展开节点比",
           title="(b) 启发函数的节省比例（越低越省）", ylim=(0, 1.05))
    a2.legend()
    fig.tight_layout()
    fig.savefig(out("e2_scale.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E3 平局规则

def e3_tie_break(seeds) -> str:
    start, goal, grid = load_grid_problem(os.path.join(ROOT, "data", "p1_sample.txt"))
    maps = [("样例 5x5", [(grid, start, goal)]),
            ("50x50 无障碍", [random_grid(50, 0.0, s) for s in seeds[:1]]),
            ("50x50 障碍20%", [random_grid(50, 0.2, s) for s in seeds])]
    rows, raw = [], []
    for label, ms in maps:
        for h in ("zero", "manhattan"):
            cells = []
            for tb in ("low_h", "fifo", "lifo"):
                exps = [solve(gr, s, g, h, tb).expanded for gr, s, g in ms]
                raw.extend([label, NAMES[h], tb, e] for e in exps)
                cells.append(mean(exps))
            rows.append([label, NAMES[h], *cells])
    write_csv("e3_tie_break.csv", ["map", "heuristic", "tie_break", "expanded"], raw)
    return "\n\n".join([
        "## E3 平局规则（f 相同时先展开谁）",
        viz.md_table(["地图", "启发函数", "low_h（h小优先）", "fifo（先入先出）", "lifo（后入先出）"], rows),
    ])


# ---------------------------------------------------------------- E4 特殊地图

def e4_special() -> str:
    rows = []
    for key, title, (grid, s, g) in (("open", "30x30 空旷地图", open_grid(30)),
                                     ("trap", "30x30 U 形陷阱", u_trap_grid(30))):
        runs = {h: solve(grid, s, g, h, trace=True) for h in ("zero", "manhattan")}
        viz.compare_expansions(grid, [(NAMES[h], r) for h, r in runs.items()],
                               out(f"e4_{key}.png"), title, start=s, goal=g)
        rows.append([title, len(runs["zero"].path) - 1, passable_cells(grid),
                     runs["zero"].expanded, runs["manhattan"].expanded,
                     runs["manhattan"].expanded / runs["zero"].expanded])
    return "\n\n".join([
        "## E4 特殊地图",
        "![](e4_open.png)", "![](e4_trap.png)",
        viz.md_table(["地图", "最短路步数", "可通行格", "展开(h=0)", "展开(曼哈顿)", "曼哈顿/h=0"], rows),
    ])


# ---------------------------------------------------------------- E5 加权 A* 与欧氏距离

WEIGHTS = (1.0, 1.5, 2.0, 3.0, 5.0)


def e5_weighted(seeds) -> str:
    configs = [("h=0", "zero", 1.0), ("欧氏", "euclidean", 1.0)] + \
              [(f"曼哈顿 w={w:g}", "manhattan", w) for w in WEIGHTS]
    rows, raw, test_rows = [], [], []
    for dens in (0.2, 0.3):
        maps = [random_grid(50, dens, s) for s in seeds]
        optimal = [len(solve(gr, s, g, "zero").path) - 1 for gr, s, g in maps]
        per_map = {}  # 配置 -> (每张图的展开数, 每张图的路径/最优)，供配对检验使用
        for label, h, w in configs:
            exps, ratios = [], []
            for (gr, s, g), opt in zip(maps, optimal):
                r = solve(gr, s, g, h, weight=w)
                exps.append(r.expanded)
                ratios.append((len(r.path) - 1) / opt)
                raw.append([dens, label, r.expanded, len(r.path) - 1, opt])
            per_map[label] = (exps, ratios)
            rows.append([f"{dens:.0%}", label, mean(exps), mean(ratios), max(ratios),
                         sum(x == 1 for x in ratios) / len(ratios)])
        # 相邻权重两两配对比较：同一批地图，看权重再加大是否还在显著地减少展开数
        for w1, w2 in zip(WEIGHTS, WEIGHTS[1:]):
            a, b = per_map[f"曼哈顿 w={w1:g}"], per_map[f"曼哈顿 w={w2:g}"]
            te, tr = paired_test(a[0], b[0]), paired_test(a[1], b[1])
            test_rows.append([f"{dens:.0%}", f"w={w1:g} → {w2:g}",
                              f"{te['diff']:+.1f} ± {te['se']:.1f}", f"{te['t']:.2f}", verdict(te),
                              f"{tr['diff']:+.3f} ± {tr['se']:.3f}", f"{tr['t']:.2f}", verdict(tr)])
    write_csv("e5_weighted.csv", ["density", "config", "expanded", "path_len", "optimal_len"], raw)
    _plot_e5(rows)

    grid, s, g = random_grid(50, 0.2, seeds[0])
    viz.compare_expansions(grid, [("欧氏", solve(grid, s, g, "euclidean", trace=True)),
                                  ("曼哈顿 w=1", solve(grid, s, g, "manhattan", trace=True)),
                                  ("曼哈顿 w=5", solve(grid, s, g, "manhattan", weight=5, trace=True))],
                           out("e5_example.png"), "50x50、障碍 20%：启发越强，搜索越集中")
    return "\n\n".join([
        f"## E5 加权 A* 与欧氏距离（50x50，每种密度 {len(seeds)} 张图）",
        "![](e5_weighted.png)", "![](e5_example.png)",
        viz.md_table(["障碍密度", "配置", "平均展开", "平均路径/最优", "最差路径/最优", "得到最优解的比例"], rows),
        "### 相邻权重的配对检验（同一批地图；差值 = 后者 − 前者，± 为标准误，|t| ≥ 2 视为显著）",
        viz.md_table(["障碍密度", "比较", "展开数差值", "t", "判断", "路径/最优 差值", "t", "判断"],
                     test_rows),
    ])


def _plot_e5(rows) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, dens in enumerate(("20%", "30%")):
        sel = [r for r in rows if r[0] == dens and r[1].startswith("曼哈顿")]
        a1.plot(WEIGHTS, [r[2] for r in sel], "o-", color=f"C{i}", label=f"障碍{dens}")
        a2.plot(WEIGHTS, [r[3] for r in sel], "o-", color=f"C{i}", label=f"障碍{dens} 平均")
        a2.plot(WEIGHTS, [r[4] for r in sel], "x:", color=f"C{i}", label=f"障碍{dens} 最差")
    a1.set(xlabel="权重 w（h' = w x 曼哈顿）", ylabel="平均展开节点数", title="(a) 搜索量")
    a2.set(xlabel="权重 w", ylabel="路径长度 / 最优长度", title="(b) 解的质量")
    a1.legend()
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("e5_weighted.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- E6 展开节点分解

def e6_f_decomposition(seeds, size=50) -> str:
    """A* 理论：启发函数一致时，f < C* 的节点一定展开，f > C* 的一定不展开，f = C* 的取决于平局规则。

    这里直接数出这两类节点，并校验实际展开情况与理论完全一致。
    """
    rows, raw = [], []
    for dens in (0.0, 0.1, 0.2, 0.3):
        acc = defaultdict(list)
        for seed in seeds:
            grid, s, g = random_grid(size, dens, seed)
            cls = classify_by_f(grid, s, g, p1_maze.cell_cost)
            cstar, must, tied = cls["cstar"], cls["must"], len(cls["tied"])
            low, fifo = solve(grid, s, g, "manhattan", trace=True), solve(grid, s, g, "manhattan", "fifo", trace=True)
            for r in (low, fifo):
                got = {st.node for st in r.steps if st.f < cstar}
                if got != must or any(st.f > cstar for st in r.steps):
                    sys.exit(f"E6 与理论不符：dens={dens} seed={seed}")
            values = dict(gap=cls["gap"], must=len(must), tied=tied, low=low.expanded,
                          fifo=fifo.expanded, zero=solve(grid, s, g, "zero").expanded, path=cstar)
            for k, v in values.items():
                acc[k].append(v)
            raw.append([dens, seed, *values.values()])
        rows.append([f"{dens:.0%}", *[mean(acc[k]) for k in ("gap", "must", "tied", "low", "fifo", "zero")],
                     mean(a / b for a, b in zip(acc["must"], acc["low"]))])
    write_csv("e6_f_decomposition.csv", ["density", "seed", "gap", "must", "tied", "low_h", "fifo", "zero", "path"], raw)
    return "\n\n".join([
        f"## E6 展开节点分解（{size}x{size}，起点左上、终点右下，每档 {len(seeds)} 张；已校验实际展开与理论一致）",
        viz.md_table(["障碍", "绕路量 C*-d", "f<C* 节点（必须展开）", "f=C* 节点（看平局）", "展开(h小优先)",
                      "展开(先进先出)", "展开(h=0)", "h小优先中必须展开的占比"], rows),
    ])


# ---------------------------------------------------------------- 入口

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="缩小规模，快速检查脚本能否跑通")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.prepare(args.quick)

    if args.quick:
        sizes, densities, seeds, repeats = (10, 20), (0.0, 0.2), range(3), 1
    else:
        sizes, densities, seeds, repeats = (10, 20, 50, 100), (0.0, 0.1, 0.2, 0.3), range(20), 3
    seeds = list(seeds)

    sections = ["# 题目一 对照实验结果", "由 `python -m experiments.exp_p1` 自动生成。"]
    corner_rows = []

    def run_e2():
        md, rows = e2_scale(sizes, densities, seeds, repeats)
        corner_rows.extend(rows)
        return md

    for name, fn in (("E1", e1_sample),
                     ("E2", run_e2),
                     ("E2b", lambda: e2b_random_endpoints(sizes, densities, seeds, repeats, corner_rows)),
                     ("E3", lambda: e3_tie_break(seeds)),
                     ("E4", e4_special),
                     ("E5", lambda: e5_weighted(seeds)),
                     ("E6", lambda: e6_f_decomposition(seeds))):
        print(f"运行 {name} ...", flush=True)
        sections.append(fn())
    print(f"完成，结果在 {OUTPUT.write_summary(sections)}")


if __name__ == "__main__":
    main()
