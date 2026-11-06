"""题目二对照实验。

运行：python -m experiments.exp_p2 [--quick]
输出：results/p2/

E1 样例：代价累计过程、h 与 h* 对照、各启发函数的展开对比     —— 报告必做
E2 可采纳性与最优性：随机加权地图上的高估、不一致、次优解统计  —— 报告必做（展开讨论）
E3 地形差异：高代价格子越多，A* 相对 Dijkstra 的优势如何变化
E4 步数最少 vs 代价最小：用题目一的做法求路径会差多少
E5 启发函数改进：统计最小代价时排除起点，最便宜地形 > 1 时效果明显
E6 渗流分析：平原格子能否连通起终点，解释 A* 优势为何在某个比例附近骤降
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

from core.astar import TIE_BREAKS
from core.grid import load_grid_problem
from experiments import viz
from experiments.admissibility import audit, true_cost_to_goal
from experiments.analysis import cheap_connected
from experiments.common import ROOT, Output, mean
from experiments.maze_gen import weighted_grid
from problems import p1_maze, p2_weighted as p2

OUTPUT = Output("p2")
out = OUTPUT.path

NAMES = {"zero": "h=0", "manhattan": "曼哈顿", "goal_aware": "修正曼哈顿",
         "manhattan_avg": "平均代价x曼哈顿"}
ALL = ("zero", "manhattan", "goal_aware", "manhattan_avg")
TERRAINS = (("平原为主", 0.2), ("混合", 0.5), ("沼泽为主", 0.8))


def heuristic(name, grid, goal, start):
    return p2.make_heuristic(name, goal, grid, start)


def path_cost(grid, path) -> float:
    return p2.cost_breakdown(grid, path)[-1][2]


def matrix_md(grid, value, mark=lambda r, c: False) -> str:
    """把每个格子的数值排成和地图同形的 Markdown 表；mark 为真的格子加粗。"""
    cols = len(grid[0])
    lines = ["| 行\\列 | " + " | ".join(map(str, range(cols))) + " |", "|---" * (cols + 1) + "|"]
    for r, row in enumerate(grid):
        cells = []
        for c, t in enumerate(row):
            v = "X" if t == "X" else f"{value(r, c):g}"
            cells.append(f"**{v}**" if t != "X" and mark(r, c) else v)
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- E1 样例

def e1_sample() -> str:
    start, goal, grid = load_grid_problem(f"{ROOT}/data/p2_sample.txt")
    res = {h: p2.solve(start, goal, grid, h, trace=True) for h in ALL}
    best = res["goal_aware"]

    # 1. 代价累计过程
    acc = [[i, p, grid[p[0]][p[1]], w, t]
           for i, (p, w, t) in enumerate(p2.cost_breakdown(grid, best.path))]

    # 2. 与"只数步数"的路径对比
    fewest = p1_maze.solve(start, goal, grid).path
    viz.draw_paths(grid, [(f"代价最小：{len(best.path) - 1} 步，代价 {best.cost:g}", best.path, "red"),
                          (f"步数最少（题目一的做法）：{len(fewest) - 1} 步，代价 {path_cost(grid, fewest):g}",
                           fewest, "royalblue")],
                   out("e1_paths.png"), "样例地形（数字为进入代价，越深越贵）", annotate=True)

    # 3. 展开过程对比
    viz.compare_expansions(grid, [(NAMES[h], res[h]) for h in ("zero", "manhattan", "goal_aware")],
                           out("e1_expansion.png"), "样例：数字为展开顺序", show_order=True)

    # 4. h 与 h* 对照
    hstar = true_cost_to_goal(grid, goal, p2.cell_cost)
    hs = {h: heuristic(h, grid, goal, start) for h in ALL}
    over = lambda h: (lambda r, c: hs[h]((r, c)) > hstar[(r, c)])
    audits = [[NAMES[h], f"{a['over_nodes']}/{a['nodes']}", a["over_max"],
               f"{a['inconsistent_edges']}/{a['edges']}", res[h].cost, res[h].expanded]
              for h in ALL for a in [audit(grid, goal, hs[h], p2.cell_cost, hstar)]]

    return "\n\n".join([
        "## E1 样例",
        "### 路径代价的累计过程",
        viz.md_table(["步", "坐标", "地形", "本步代价", "累计代价"], acc),
        "![](e1_paths.png)",
        "### 每个格子的真实剩余代价 h*（反向 Dijkstra 求得）",
        matrix_md(grid, lambda r, c: hstar[(r, c)]),
        "### 曼哈顿距离 h（加粗 = 高估，即 h > h*）",
        matrix_md(grid, lambda r, c: hs["manhattan"]((r, c)), over("manhattan")),
        "### 修正曼哈顿 h = (d-1) x 最小代价 + 终点代价（加粗 = 高估）",
        matrix_md(grid, lambda r, c: hs["goal_aware"]((r, c)), over("goal_aware")),
        "### 各启发函数的性质与结果",
        viz.md_table(["启发函数", "高估格子", "最大高估量", "不一致的边", "求得代价", "展开节点"], audits),
        "![](e1_expansion.png)",
        "### 修正曼哈顿的逐步过程",
        viz.step_table_md(best),
    ])


# ---------------------------------------------------------------- E2 可采纳性与最优性

def e2_admissibility(seeds, size) -> str:
    raw = []
    for terrain, p in TERRAINS:
        for seed in seeds:
            grid, s, g = weighted_grid(size, seed, p_high=p)
            hstar = true_cost_to_goal(grid, g, p2.cell_cost)
            opt = hstar[s]
            for h in ALL:
                a = audit(grid, g, heuristic(h, grid, g, s), p2.cell_cost, hstar)
                r = p2.solve(s, g, grid, h)
                if h in ("zero", "goal_aware") and r.cost != opt:
                    sys.exit(f"可采纳启发函数得到次优解：{terrain} seed={seed} {h}")
                raw.append([terrain, seed, h, a["over_nodes"] / a["nodes"], a["over_max"],
                            a["inconsistent_edges"] / a["edges"], r.expanded, r.reopened, r.cost, opt])
    OUTPUT.write_csv("e2_admissibility.csv",
                     ["terrain", "seed", "heuristic", "over_frac", "over_max", "inconsistent_frac",
                      "expanded", "reopened", "cost", "optimal"], raw)

    groups = defaultdict(list)
    for row in raw:
        groups[(row[0], row[2])].append(row)
    rows = []
    for terrain, _ in TERRAINS:
        for h in ALL:
            rs = groups[(terrain, h)]
            excess = [r[8] / r[9] - 1 for r in rs]
            rows.append([terrain, NAMES[h], f"{mean(r[3] for r in rs):.1%}", max(r[4] for r in rs),
                         f"{mean(r[5] for r in rs):.2%}", mean(r[6] for r in rs), mean(r[7] for r in rs),
                         f"{sum(e > 0 for e in excess) / len(excess):.0%}", f"{max(excess):.1%}"])
    return "\n\n".join([
        f"## E2 可采纳性与最优性（{size}x{size}，每种地形 {len(seeds)} 张图）",
        viz.md_table(["地形", "启发函数", "高估格子比例", "最大高估量", "不一致边比例", "平均展开",
                      "平均重新打开", "次优解比例", "最大超出最优"], rows),
    ])


def e2b_bulk(n_maps) -> str:
    """大批量小地图 x 三种平局规则，检验"曼哈顿虽不可采纳却总能得到最优解"是否普遍成立。"""
    rng = random.Random(2026)
    stats = defaultdict(lambda: [0, 0, 0.0])  # (h, tb) -> [地图数, 次优次数, 最大超出]
    fewer = 0  # 曼哈顿展开数 <= 修正曼哈顿 的次数
    for i in range(n_maps):
        size = rng.randint(4, 15)
        grid, s, g = weighted_grid(size, i, p_high=rng.uniform(0.1, 0.9),
                                   density=rng.choice((0.0, 0.1, 0.2)))
        opt = p2.solve(s, g, grid, "zero").cost
        for tb in TIE_BREAKS:
            exp = {}
            for h in ("manhattan", "goal_aware", "manhattan_avg"):
                r = p2.solve(s, g, grid, h, tie_break=tb)
                st = stats[(h, tb)]
                st[0] += 1
                st[1] += r.cost != opt
                st[2] = max(st[2], r.cost / opt - 1)
                exp[h] = r.expanded
            fewer += exp["manhattan"] <= exp["goal_aware"]
    rows = [[NAMES[h], tb, n, k, f"{k / n:.1%}", f"{mx:.1%}"] for (h, tb), (n, k, mx) in stats.items()]
    return "\n\n".join([
        f"## E2b 大批量最优性校验（{n_maps} 张 4~15 边长的随机地图，含障碍）",
        viz.md_table(["启发函数", "平局规则", "求解次数", "次优次数", "次优比例", "最大超出最优"], rows),
        f"曼哈顿的展开数不多于修正曼哈顿的情况：{fewer}/{n_maps * len(TIE_BREAKS)}。",
    ])


# ---------------------------------------------------------------- E3 地形差异

P_SWEEP = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)


def e3_terrain(seeds, size) -> str:
    raw = []
    for p in P_SWEEP:
        for seed in seeds:
            grid, s, g = weighted_grid(size, seed, p_high=p)
            hstar_s = p2.solve(s, g, grid, "zero")
            ga = p2.solve(s, g, grid, "goal_aware")
            informed = heuristic("goal_aware", grid, g, s)(s) / hstar_s.cost
            raw.append([p, seed, hstar_s.cost, hstar_s.expanded, ga.expanded, informed])
    OUTPUT.write_csv("e3_terrain.csv", ["p_high", "seed", "optimal", "expanded_zero",
                                         "expanded_goal_aware", "h_over_hstar_at_start"], raw)
    rows = []
    for p in P_SWEEP:
        rs = [r for r in raw if r[0] == p]
        rows.append([f"{p:.0%}", mean(r[2] for r in rs), mean(r[3] for r in rs), mean(r[4] for r in rs),
                     mean(r[4] / r[3] for r in rs), mean(r[5] for r in rs)])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [p * 100 for p in P_SWEEP]
    a1.plot(xs, [r[2] for r in rows], "o-", label="h=0（Dijkstra）")
    a1.plot(xs, [r[3] for r in rows], "s--", label="修正曼哈顿")
    a1.set(xlabel="高代价格子比例 (%)", ylabel="平均展开节点数", title="(a) 搜索量")
    a1.legend()
    a2.plot(xs, [r[4] for r in rows], "o-", color="C2", label="展开数之比（修正曼哈顿 / h=0）")
    a2.plot(xs, [r[5] for r in rows], "s--", color="C3", label="起点处 h / h*（启发的信息量）")
    a2.set(xlabel="高代价格子比例 (%)", ylim=(0, 1.05), title="(b) 启发越不准，节省越少")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("e3_terrain.png"), dpi=150)
    plt.close(fig)

    grid, s, g = weighted_grid(40, seeds[0], p_high=0.5)
    viz.compare_expansions(grid, [(NAMES[h], p2.solve(s, g, grid, h, trace=True))
                                  for h in ("zero", "goal_aware")],
                           out("e3_example.png"), "40x40、高代价格子 50%（底图越深代价越高）")
    return "\n\n".join([
        f"## E3 地形差异（{size}x{size}，每档 {len(seeds)} 张图）",
        "![](e3_terrain.png)", "![](e3_example.png)",
        viz.md_table(["高代价比例", "平均最优代价", "展开(h=0)", "展开(修正曼哈顿)", "比值", "起点 h/h*"], rows),
    ])


# ---------------------------------------------------------------- E4 步数最少 vs 代价最小

def e4_steps_vs_cost(seeds, size) -> str:
    rows = []
    for p in (0.3, 0.6):
        rs = []
        for seed in seeds:
            grid, s, g = weighted_grid(size, seed, p_high=p)
            best = p2.solve(s, g, grid)
            fewest = p1_maze.solve(s, g, grid).path
            rs.append([best.cost, path_cost(grid, fewest), len(best.path) - 1, len(fewest) - 1])
        rows.append([f"{p:.0%}", mean(r[0] for r in rs), mean(r[1] for r in rs),
                     mean(r[1] / r[0] for r in rs), mean(r[2] for r in rs), mean(r[3] for r in rs)])

    grid, s, g = weighted_grid(size, seeds[0], p_high=0.5)
    best, fewest = p2.solve(s, g, grid).path, p1_maze.solve(s, g, grid).path
    viz.draw_paths(grid, [(f"代价最小：{len(best) - 1} 步，代价 {path_cost(grid, best):g}", best, "red"),
                          (f"步数最少：{len(fewest) - 1} 步，代价 {path_cost(grid, fewest):g}",
                           fewest, "royalblue")],
                   out("e4_paths.png"), f"{size}x{size}、高代价格子 50%")
    return "\n\n".join([
        f"## E4 步数最少 vs 代价最小（{size}x{size}，每档 {len(seeds)} 张图）",
        "![](e4_paths.png)",
        viz.md_table(["高代价比例", "最优代价", "步数最少路径的代价", "代价之比", "最优路径步数",
                      "最少步数"], rows),
    ])


# ---------------------------------------------------------------- E5 启发函数改进：w_min 排除起点

LOW_COSTS = ((1, (3, 8)), (2, (5, 9)), (3, (5, 9)))


def e5_exclude_start(seeds, size) -> str:
    """最便宜的地形代价 > 1 时，起点 S 记的 1 会把 w_min 拉低。比较改进前后。"""
    configs = (("h=0（Dijkstra）", "zero"), ("改进前：w_min 含起点", "goal_aware_incl_start"),
               ("改进后：w_min 排除起点", "goal_aware"))
    rows, raw = [], []
    for low, high in LOW_COSTS:
        acc = {name: ([], [], []) for name, _ in configs}
        for seed in seeds:
            grid, s, g = weighted_grid(size, seed, p_high=0.5, low=low, high=high)
            hstar = true_cost_to_goal(grid, g, p2.cell_cost, blocked=(s,))
            opt = p2.solve(s, g, grid, "zero").cost
            for name, h in configs:
                r = p2.solve(s, g, grid, h)
                a = audit(grid, g, heuristic(h, grid, g, s), p2.cell_cost, hstar, blocked=(s,))
                if r.cost != opt:
                    sys.exit(f"E5 得到次优解：low={low} seed={seed} {h}")
                acc[name][0].append(r.expanded)
                acc[name][1].append(a["over_nodes"])
                acc[name][2].append(heuristic(h, grid, g, s)(s) / opt)
                raw.append([low, seed, h, r.expanded, a["over_nodes"], r.cost, opt])
        for name, _ in configs:
            exp, over, inf = acc[name]
            rows.append([f"{low}（高代价取 {high}）", name, mean(exp), max(over), mean(inf)])
    OUTPUT.write_csv("e5_exclude_start.csv",
                     ["low", "seed", "heuristic", "expanded", "over_nodes", "cost", "optimal"], raw)
    return "\n\n".join([
        f"## E5 启发函数改进：统计最小代价时排除起点（{size}x{size}，高代价格子 50%，每档 {len(seeds)} 张图）",
        "高估格子按\"不经过起点的路径\"计算真实剩余代价：从起点出发的搜索，剩余路径不会再经过起点。",
        viz.md_table(["最便宜的地形代价", "启发函数", "平均展开", "高估格子（最大）", "起点处 h/h*"], rows),
    ])


# ---------------------------------------------------------------- E6 渗流分析

P_FINE = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.9)


def e6_percolation(seeds, size) -> str:
    """h 只知道"每步至少花 1"。只要存在一条全由平原组成的不绕路路线，h 就几乎等于 h*；
    这条路线一旦消失，真实代价陡增，h 的信息量随之骤降。这是一种渗流现象：
    二维方格上"定向渗流"（只能向下向右）的开放概率阈值约 0.705，"普通渗流"约 0.593。"""
    rows = []
    for p in P_FINE:
        rs = []
        for seed in seeds:
            grid, s, g = weighted_grid(size, seed, p_high=p)
            z, a = p2.solve(s, g, grid, "zero"), p2.solve(s, g, grid, "goal_aware")
            rs.append([cheap_connected(grid, s, g, True), cheap_connected(grid, s, g, False),
                       heuristic("goal_aware", grid, g, s)(s) / z.cost, a.expanded / z.expanded])
        rows.append([f"{p:.0%}", f"{1 - p:.0%}", *[mean(r[i] for r in rs) for i in range(4)]])
    xs = [float(r[0].rstrip("%")) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (label, style) in enumerate((("存在不绕路的全平原路线的比例", "s--"), ("存在全平原路线的比例（可绕路）", "^--"),
                                        ("起点处 h / h*", "o-"), ("展开数之比 A* / Dijkstra", "D-"))):
        ax.plot(xs, [r[2 + i] for r in rows], style, label=label)
    ax.set(xlabel="高代价格子比例 (%)", ylim=(-0.02, 1.05), title=f"{size}x{size} 加权地形，每档 {len(seeds)} 张")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out("e6_percolation.png"), dpi=150)
    plt.close(fig)
    return "\n\n".join([
        f"## E6 渗流分析（{size}x{size}，每档 {len(seeds)} 张）",
        "![](e6_percolation.png)",
        viz.md_table(["高代价比例", "平原比例", "存在不绕路全平原路线", "存在全平原路线", "起点 h/h*", "A*/Dijkstra 展开比"], rows),
    ])


# ---------------------------------------------------------------- 入口

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="缩小规模，快速检查脚本能否跑通")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.prepare(args.quick)
    seeds = list(range(3 if args.quick else 20))
    size = 20 if args.quick else 50
    n_bulk = 100 if args.quick else 2000

    sections = ["# 题目二 对照实验结果", "由 `python -m experiments.exp_p2` 自动生成。"]
    for name, fn in (("E1", e1_sample),
                     ("E2", lambda: e2_admissibility(seeds, size)),
                     ("E2b", lambda: e2b_bulk(n_bulk)),
                     ("E3", lambda: e3_terrain(seeds, size)),
                     ("E4", lambda: e4_steps_vs_cost(seeds, 30)),
                     ("E5", lambda: e5_exclude_start(seeds, size)),
                     ("E6", lambda: e6_percolation(seeds, size))):
        print(f"运行 {name} ...", flush=True)
        sections.append(fn())
    print(f"完成，结果在 {OUTPUT.write_summary(sections)}")


if __name__ == "__main__":
    main()
