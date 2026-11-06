"""题目三对照实验。

运行：python -m experiments.exp_p3 [--quick]
输出：results/p3/

E1 样例：Minimax 回传值、Alpha-Beta 逐节点窗口与剪枝、四种访问顺序   —— 报告必做
E2 随机博弈树：分支数/深度 x 访问顺序，与理论值 b^d、Knuth-Moore 公式对照 —— 报告必做（访问顺序）
E3 启发式排序：用带噪声的估值排序，排序越准剪得越多                       —— 拓展（节点排序）
E4 剪枝条件：alpha >= beta 与 alpha > beta                              —— 拓展
E5 置换表：共享节点/置换很多时，缓存能省多少                             —— 拓展（缓存）
E6 剪枝发生的位置：访问顺序为什么影响开销                                —— 分析
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from core.game_search import alphabeta, minimax
from experiments import viz
from experiments.common import ROOT, Output, mean
from experiments.game_trees import (correlated_tree, exact_values, first_move_accuracy,
                                    knuth_moore_leaves, layered_dag, noisy_estimates,
                                    order_best, order_by_estimate, order_reverse, order_worst,
                                    random_tree)
from problems import p3_tree

OUTPUT = Output("p3")
out = OUTPUT.path
GAME = p3_tree.TreeGame()


def search(root, method="alphabeta", **kw):
    algo = {"alphabeta": alphabeta, "minimax": minimax}[method]
    return algo(GAME, GAME.initial_state(root), **kw)


def orders_for(root) -> dict:
    """四种访问顺序。理想/最差顺序借助预先算好的真实值，排序本身的开销不计入。"""
    v = exact_values(root)
    return {"从左到右": None, "从右到左": order_reverse, "理想顺序": order_best(v), "最差顺序": order_worst(v)}


def check(results, expected) -> None:
    for r in results:
        if r.value != expected:
            sys.exit(f"结果不一致：{r.value} != {expected}")


def path_name(path) -> str:
    return "-".join(["root", *map(str, path)])


# ---------------------------------------------------------------- E1 样例

def e1_sample() -> str:
    root = p3_tree.build_sample_tree()
    mm = search(root, "minimax")
    ab = search(root, trace=True)
    viz.save_game_trees(root, [("Minimax：每个节点的回传值", None),
                               ("Alpha-Beta（从左到右）：#访问序号 [进入时的 α, β]", ab.stats.trace)],
                        out("e1_tree.png"))

    trace_rows = []
    for t in ab.stats.trace:
        kind = "叶子" if t.leaf else ("MAX" if t.is_max else "MIN")
        if t.leaf:
            note = ""
        elif t.cut:
            bound = "下界" if t.is_max else "上界"
            note = f"α={t.value if t.is_max else t.alpha:g} ≥ β={t.beta if t.is_max else t.value:g}，" \
                   f"剪掉子节点 {t.cut}；返回值只是{bound}"
        else:
            note = "子节点全部访问"
        trace_rows.append([t.index, path_name(t.path), kind, viz.fmt_num(t.alpha), viz.fmt_num(t.beta),
                           ("≥" if t.is_max else "≤") + f"{t.value:g}" if t.cut else f"{t.value:g}", note])

    panels, rows = [], []
    for name, order in orders_for(root).items():
        r = search(root, order=order, trace=True)
        check([r], mm.value)
        rows.append([name, r.value, r.stats.nodes, r.stats.leaves, r.stats.cutoffs])
        panels.append((f"{name}：访问 {r.stats.nodes} 个节点，评估 {r.stats.leaves} 片叶子", r.stats.trace))
    rows.insert(0, ["Minimax（不剪枝）", mm.value, mm.stats.nodes, mm.stats.leaves, 0])
    viz.save_game_trees(root, panels, out("e1_orders.png"), "样例在四种访问顺序下的剪枝情况")

    return "\n\n".join([
        "## E1 样例",
        "![](e1_tree.png)",
        "### Alpha-Beta 逐节点过程（从左到右，先序）",
        viz.md_table(["序号", "节点", "类型", "进入时 α", "进入时 β", "返回值", "说明"], trace_rows),
        "### 四种访问顺序",
        viz.md_table(["方法 / 访问顺序", "根节点值", "访问节点", "评估叶子", "剪枝次数"], rows),
        "![](e1_orders.png)",
    ])


# ---------------------------------------------------------------- E2 随机博弈树

ORDER_KEYS = ("理想顺序", "从左到右", "最差顺序")


def _run_random(b, d, seeds):
    acc = {k: [] for k in ("Minimax", *ORDER_KEYS)}
    times = {"Minimax": [], "从左到右": []}
    for seed in seeds:
        root = random_tree(b, d, seed)
        mm = search(root, "minimax")
        acc["Minimax"].append(mm.stats.leaves)
        times["Minimax"].append(mm.stats.elapsed)
        orders = orders_for(root)
        for k in ORDER_KEYS:
            r = search(root, order=orders[k])
            check([r], mm.value)
            acc[k].append(r.stats.leaves)
            if k == "从左到右":
                times[k].append(r.stats.elapsed)
    return {k: mean(v) for k, v in acc.items()}, {k: mean(v) * 1000 for k, v in times.items()}


def e2_random(seeds, depths, branchings) -> str:
    raw, rows = [], []
    for sweep, pairs in (("深度", [(3, d) for d in depths]), ("分支数", [(b, 5) for b in branchings])):
        for b, d in pairs:
            leaves, ms = _run_random(b, d, seeds)
            km = knuth_moore_leaves(b, d)
            raw.append([sweep, b, d, km, *[leaves[k] for k in ("Minimax", *ORDER_KEYS)]])
            rows.append([sweep, b, d, b ** d, km, leaves["理想顺序"], leaves["从左到右"], leaves["最差顺序"],
                         leaves["从左到右"] ** (1 / d), ms["Minimax"], ms["从左到右"]])
    OUTPUT.write_csv("e2_random.csv", ["sweep", "b", "d", "knuth_moore", "minimax", *ORDER_KEYS], raw)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, sweep, xkey, xlabel in ((axes[0], "深度", 2, "深度 d（b=3）"), (axes[1], "分支数", 1, "分支数 b（d=5）")):
        sel = [r for r in raw if r[0] == sweep]
        xs = [r[xkey] for r in sel]
        ax.plot(xs, [r[4] for r in sel], "k-", lw=2, label="Minimax = b^d")
        ax.plot(xs, [r[7] for r in sel], "^-", color="C3", label="最差顺序")
        ax.plot(xs, [r[6] for r in sel], "o-", color="C0", label="从左到右（随机）")
        ax.plot(xs, [r[5] for r in sel], "s-", color="C2", label="理想顺序")
        ax.plot(xs, [r[3] for r in sel], ":", color="C2", lw=2.5, label="Knuth-Moore 理论值")
        ax.set(yscale="log", xlabel=xlabel, ylabel="评估的叶子数（对数坐标）")
        ax.legend(fontsize=8)
    axes[0].set_title("(a) 固定 b=3，改变深度")
    axes[1].set_title("(b) 固定 d=5，改变分支数")
    fig.tight_layout()
    fig.savefig(out("e2_random.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        f"## E2 随机博弈树（叶子为 [0,1) 随机实数，每组 {len(seeds)} 棵树取平均）",
        "![](e2_random.png)",
        viz.md_table(["扫描", "b", "d", "Minimax 叶子 b^d", "Knuth-Moore 理论", "理想顺序", "从左到右",
                      "最差顺序", "从左到右的有效分支数", "Minimax 耗时ms", "Alpha-Beta 耗时ms"], rows),
    ])


# ---------------------------------------------------------------- E3 启发式排序

TAUS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def e3_heuristic_order(seeds, b=4, d=6) -> str:
    acc = {k: [] for k in ("从左到右", "理想顺序", "最差顺序")}
    by_tau = {t: ([], []) for t in TAUS}
    for seed in seeds:
        root = correlated_tree(b, d, seed)
        values = exact_values(root)
        orders = orders_for(root)
        for k in acc:
            acc[k].append(search(root, order=orders[k]).stats.leaves)
        for tau in TAUS:
            est = noisy_estimates(root, tau, seed)
            r = search(root, order=order_by_estimate(est))
            check([r], values[(id(root), True)])
            by_tau[tau][0].append(r.stats.leaves)
            by_tau[tau][1].append(first_move_accuracy(root, values, est))
    base = {k: mean(v) for k, v in acc.items()}
    rows = [[f"估值排序 τ={t:g}", mean(by_tau[t][0]), f"{mean(by_tau[t][1]):.0%}"] for t in TAUS]
    rows = [["理想顺序", base["理想顺序"], "100%"], *rows, ["从左到右（不排序）", base["从左到右"], "-"],
            ["最差顺序", base["最差顺序"], "-"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(TAUS, [mean(by_tau[t][0]) for t in TAUS], "o-", color="C1", label="按估值排序")
    for k, c in (("理想顺序", "C2"), ("从左到右", "C0"), ("最差顺序", "C3")):
        ax.axhline(base[k], color=c, ls="--", label=k)
    ax2 = ax.twinx()
    ax2.plot(TAUS, [mean(by_tau[t][1]) for t in TAUS], "s:", color="0.4", label="首选命中率（右轴）")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("排在第一的子节点恰为最优的比例")
    ax.set(xlabel="估值噪声 τ（越大估值越不准）", ylabel="平均评估叶子数",
           title=f"相关博弈树 b={b}, d={d}：排序质量与剪枝效果")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out("e3_heuristic_order.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        f"## E3 启发式排序（相关博弈树 b={b}, d={d}，{len(seeds)} 棵树取平均；Minimax 需评估 {b ** d} 片叶子）",
        "![](e3_heuristic_order.png)",
        viz.md_table(["访问顺序", "平均评估叶子", "首选命中率"], rows),
    ])


# ---------------------------------------------------------------- E4 剪枝条件

def e4_prune_condition(seeds, b=3, d=6) -> str:
    rows = []
    for label, vals in (("实数（几乎无相等）", None), ("整数 0~9", (0, 9)), ("整数 0~2（大量相等）", (0, 2))):
        ge, gt = [], []
        for seed in seeds:
            root = random_tree(b, d, seed, int_values=vals)
            r1, r2 = search(root), search(root, prune_equal=False)
            check([r2], r1.value)
            ge.append(r1.stats.leaves)
            gt.append(r2.stats.leaves)
        rows.append([label, mean(ge), mean(gt), mean(g / t for g, t in zip(ge, gt))])
    return "\n\n".join([
        f"## E4 剪枝条件（b={b}, d={d}，从左到右，{len(seeds)} 棵树）",
        viz.md_table(["叶子取值", "α≥β 时剪（叶子数）", "α>β 时剪（叶子数）", "比值"], rows),
    ])


# ---------------------------------------------------------------- E5 置换表

def e5_cache(seeds, depths, width=6, b=3) -> str:
    sample_rows = []
    for label, root in (("TreeNode 版（max2 共享）", p3_tree.build_sample_tree()),
                        ("JSON 版（max2 复制两份）", p3_tree.load_tree(f"{ROOT}/data/p3_sample.json"))):
        for name, kw in (("Minimax", dict(method="minimax")), ("Minimax+缓存", dict(method="minimax", cache=True)),
                         ("Alpha-Beta", {}), ("Alpha-Beta+置换表", dict(cache=True))):
            r = search(root, **kw)
            sample_rows.append([label, name, r.value, r.stats.nodes, r.stats.leaves, r.stats.tt_hits])

    configs = (("Minimax", dict(method="minimax")), ("Minimax+缓存", dict(method="minimax", cache=True)),
               ("Alpha-Beta", {}), ("Alpha-Beta+置换表", dict(cache=True)))
    rows, series = [], {name: [] for name, _ in configs}
    for d in depths:
        acc = {name: [] for name, _ in configs}
        for seed in seeds:
            root = layered_dag(d, width, b, seed)
            results = {name: search(root, **kw) for name, kw in configs}
            check(results.values(), results["Minimax+缓存"].value)
            for name, r in results.items():
                acc[name].append(r.stats.nodes)
        for name in acc:
            series[name].append(mean(acc[name]))
        rows.append([d, d * width + 1, *[mean(acc[name]) for name, _ in configs]])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, _ in configs:
        ax.plot(depths, series[name], "o-", label=name)
    ax.plot(depths, [r[1] for r in rows], "k:", label="不同节点总数")
    ax.set(yscale="log", xlabel="深度", ylabel="访问节点数（对数坐标）",
           title=f"分层 DAG（每层 {width} 个节点，分支数 {b}）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("e5_cache.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        "## E5 置换表 / 缓存",
        "### 样例",
        viz.md_table(["输入", "方法", "根节点值", "访问节点", "评估叶子", "缓存命中"], sample_rows),
        f"### 分层 DAG（大量共享节点，每个深度 {len(seeds)} 个随机图）",
        "![](e5_cache.png)",
        viz.md_table(["深度", "不同节点数", *[name for name, _ in configs]], rows),
    ])


# ---------------------------------------------------------------- E6 剪枝发生的位置

def _first_child_share(res, b: int) -> float:
    """发生剪枝的节点中，看完第一个子节点就剪枝的比例（满 b 叉树：剪掉了其余 b-1 个）。"""
    cuts = res.stats.pruned
    return sum(len(c) == b - 1 for _, c in cuts) / len(cuts) if cuts else 0.0


def e6_cut_position(seeds, b=4, d=6) -> str:
    """访问顺序好不好，直接体现在剪枝发生得早不早：
    理想顺序下每次剪枝都发生在第一个子节点，于是每隔一层只需看一个孩子，叶子数降到约 b^(d/2)。"""
    configs = {"理想顺序": [], "估值排序 τ=0": [], "估值排序 τ=1": [], "估值排序 τ=4": [], "从左到右": [], "最差顺序": []}
    for seed in seeds:
        root = correlated_tree(b, d, seed)
        orders = orders_for(root)
        orders.update({f"估值排序 τ={t}": order_by_estimate(noisy_estimates(root, t, seed)) for t in (0, 1, 4)})
        for name, stats in configs.items():
            r = search(root, order=orders[name])
            stats.append((r.stats.leaves, r.stats.cutoffs, _first_child_share(r, b)))
    rows = [[name, mean(x[0] for x in st), mean(x[1] for x in st), f"{mean(x[2] for x in st):.0%}"]
            for name, st in configs.items()]
    return "\n\n".join([
        f"## E6 剪枝发生的位置（相关博弈树 b={b}, d={d}，{len(seeds)} 棵）",
        viz.md_table(["访问顺序", "评估叶子", "剪枝次数", "看完第一个子节点就剪枝的比例"], rows),
    ])


# ---------------------------------------------------------------- 入口

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="缩小规模，快速检查脚本能否跑通")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.prepare(args.quick)
    if args.quick:
        seeds, depths, branchings, dag_depths = list(range(2)), (2, 3, 4), (2, 3), (4, 6)
    else:
        seeds, depths, branchings, dag_depths = list(range(10)), tuple(range(2, 10)), tuple(range(2, 9)), (4, 6, 8, 10)

    sections = ["# 题目三 对照实验结果", "由 `python -m experiments.exp_p3` 自动生成。"]
    for name, fn in (("E1", e1_sample),
                     ("E2", lambda: e2_random(seeds, depths, branchings)),
                     ("E3", lambda: e3_heuristic_order(list(range(len(seeds) * 2)))),
                     ("E4", lambda: e4_prune_condition(list(range(len(seeds) * 2)))),
                     ("E5", lambda: e5_cache(seeds[:5], dag_depths)),
                     ("E6", lambda: e6_cut_position(list(range(len(seeds) * 2))))):
        print(f"运行 {name} ...", flush=True)
        sections.append(fn())
    print(f"完成，结果在 {OUTPUT.write_summary(sections)}")


if __name__ == "__main__":
    main()
