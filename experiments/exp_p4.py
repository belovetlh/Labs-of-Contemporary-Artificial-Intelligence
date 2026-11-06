"""题目四对照实验。

运行：python -m experiments.exp_p4 [--quick]
输出：results/p4/

E1 样例：距离矩阵与最短路径、状态表示、各深度下每个首步的价值、博弈树剪枝图  —— 报告必做
E2 访问顺序：自然/反序/贪心/理想/最差 五种顺序在 H=2、H=4 下的剪枝效果         —— 报告必做
E3 状态设计与缓存：去掉累计收益的置换表键；宝藏数 4~10 的随机地图上的规模增长   —— 拓展（缓存/排序）
E4 视野效应：A 用不同深度实际走完整局，最终收益差如何                           —— 拓展（深度的影响）
E5 剪枝空间：分支数逐层递减时，完美排序下的最少叶子数                           —— 分析
E6 损失分解与主变例对比：损失来自哪一步决策，T3 为什么好                        —— 分析
"""
from __future__ import annotations

import argparse
import sys
import time

import matplotlib.pyplot as plt

from experiments import viz
from experiments.analysis import min_leaves
from experiments.common import ROOT, Output, mean, paired_test, verdict
from experiments.maze_gen import treasure_map
from problems import p4_treasure as p4
from problems.p3_tree import TreeGame, TreeNode
from core.game_search import alphabeta

OUTPUT = Output("p4")
out = OUTPUT.path


def build_game(grid, values):
    dist, _ = p4.distance_matrix(grid, p4.locate_points(grid))
    return p4.TreasureGame(dist, values)


def load_sample():
    grid, values, _ = p4.load(f"{ROOT}/data/p4_sample.json")
    return grid, values, build_game(grid, values)


def to_tree(game, depth, order=None):
    """把深度为 depth 的博弈树展开成 TreeNode，便于用题目三的画图工具展示。

    返回 (根节点, 边标签)。子节点顺序与搜索时的访问顺序一致，因此剪枝情况完全相同。
    """
    labels = {}

    def build(s, d, path):
        acts = [] if d == 0 else game.actions(s)
        if order and acts:
            acts = order(game, s, acts)
        node = TreeNode(value=s.utility if not acts else None)
        node.is_max = s.turn == "A"
        for i, t in enumerate(acts):
            labels[path + (i,)] = t  # 走法方由父节点类型可知：MAX 为 A，MIN 为 B
            node.children.append(build(game.result(s, t), d - 1, path + (i,)))
        return node
    return build(game.initial_state(), depth, ()), labels


def check(results, expected) -> None:
    for r in results:
        if r.value != expected:
            sys.exit(f"结果不一致：{r.value} != {expected}")


# ---------------------------------------------------------------- E1 样例

def e1_sample() -> str:
    grid, values, game = load_sample()
    points = p4.locate_points(grid)
    names = list(points)

    # 1. 距离矩阵与 A* 开销
    runs = {h: p4.pairwise_astar(grid, points, h) for h in ("manhattan", "zero")}
    dist = game.dist
    matrix = [[a, *[dist[a][b] for b in names]] for a in names]
    sym = all(dist[a][b] == dist[b][a] for a in names for b in names)
    astar_rows = [[label, len(r), sum(x.expanded for x in r.values())]
                  for label, r in (("曼哈顿 x 最小代价", runs["manhattan"]), ("h=0（Dijkstra）", runs["zero"]))]
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]
    treasures = names[2:]
    panels = [(f"{who} 到各宝藏的最短路径",
               [(f"{who}→{t}：代价 {dist[who][t]}", runs["manhattan"][(who, t)].path, c)
                for t, c in zip(treasures, colors)]) for who in ("A", "B")]
    viz.save_path_panels(grid, panels, out("e1_paths.png"), "A* 求得的最短路径（格中数字为进入代价）")

    # 2. 状态表示
    s0 = game.initial_state()
    s1 = game.result(s0, "T2")
    fmt = lambda s: [s.pos_a, s.pos_b, "{" + ",".join(sorted(s.remaining, key=p4.point_key)) + "}",
                     s.score_a, s.score_b, s.turn, s.utility]
    state_rows = [["初始状态", *fmt(s0)], ["A 选 T2 之后", *fmt(s1)]]

    # 3. 每个深度下各首步的价值
    move_rows, pv_rows = [], []
    for h in (1, 2, 3, 4):
        vals = {t: p4.solve_incremental(game, h - 1, state=game.result(s0, t)).value for t in treasures}
        best = max(vals.values())
        move_rows.append([f"H={h}", *[f"**{v}**" if v == best else v for v in vals.values()]])
        res = p4.solve(game, h)
        pv_rows.append([f"H={h}", res.best_action, res.value, p4.explain_pv(game, res.pv)])

    # 4. Minimax 与 Alpha-Beta 在 H=2、H=4 的开销
    cmp_rows = []
    for h in (2, 4):
        for method in ("minimax", "alphabeta"):
            r = p4.solve(game, h, method)
            cmp_rows.append([f"H={h}", method, r.best_action, r.value, r.stats.nodes, r.stats.leaves,
                             r.stats.cutoffs])

    # 5. 博弈树图
    for h, size in ((2, (8, 4.2)), (4, (22, 5.5))):
        root, labels = to_tree(game, h)
        trace = alphabeta(TreeGame(), TreeGame().initial_state(root), trace=True).stats.trace
        panels = [(f"Alpha-Beta（H={h}，访问顺序 T1→T4）", trace)]
        if h == 2:
            panels.insert(0, ("Minimax（H=2）：每个节点的回传值", None))
        viz.save_game_trees(root, panels, out(f"e1_tree_h{h}.png"), ncols=2 if h == 2 else 1,
                            panel_size=size, edge_labels=labels)

    return "\n\n".join([
        "## E1 样例",
        "### 距离矩阵（A* 计算，行 = 出发点，列 = 目标点）",
        viz.md_table(["", *names], matrix),
        f"矩阵{'对称' if sym else '不对称'}：A、B、T 所在格进入代价都是 1，所以 d(p,q) = d(q,p)。",
        viz.md_table(["启发函数", "A* 调用次数", "总展开节点"], astar_rows),
        "![](e1_paths.png)",
        "### 状态表示",
        viz.md_table(["", "pos_a", "pos_b", "remaining", "score_a", "score_b", "turn", "U"], state_rows),
        "### 各深度下每个首步的价值（加粗为最优）",
        viz.md_table(["深度", *treasures], move_rows),
        viz.md_table(["深度", "最优首步", "最优收益差", "主变例"], pv_rows),
        "### Minimax 与 Alpha-Beta（访问顺序 T1→T4）",
        viz.md_table(["深度", "方法", "最优首步", "最优收益差", "访问节点", "评估叶子", "剪枝次数"], cmp_rows),
        "![](e1_tree_h2.png)", "![](e1_tree_h4.png)",
    ])


# ---------------------------------------------------------------- E2 访问顺序

def e2_orders() -> str:
    _, _, game = load_sample()
    rows = []
    for h in (2, 4):
        orders = {"自然（T1→T4）": None, "反序（T4→T1）": p4.order_reverse, "贪心（即时收益高者先）": p4.order_greedy,
                  "理想": p4.order_oracle(game, h), "最差": p4.order_oracle(game, h, worst=True)}
        mm = p4.solve(game, h, "minimax")
        rows.append([f"H={h}", "Minimax（不剪枝）", mm.best_action, mm.value, mm.stats.nodes, mm.stats.leaves, 0])
        for name, order in orders.items():
            r = p4.solve(game, h, order=order)
            check([r], mm.value)
            rows.append([f"H={h}", name, r.best_action, r.value, r.stats.nodes, r.stats.leaves, r.stats.cutoffs])

    for i, (name, order) in enumerate((("自然顺序 T1→T4", None), ("贪心顺序", p4.order_greedy))):
        root, labels = to_tree(game, 4, order)
        r = alphabeta(TreeGame(), TreeGame().initial_state(root), trace=True)
        title = f"H=4，{name}：访问 {r.stats.nodes} 个节点，剪枝 {r.stats.cutoffs} 次"
        viz.save_game_trees(root, [(title, r.stats.trace)], out(f"e2_tree_{i}.png"), ncols=1,
                            panel_size=(22, 5.5), edge_labels=labels)
    return "\n\n".join([
        "## E2 访问顺序与剪枝",
        viz.md_table(["深度", "访问顺序", "最优首步", "最优收益差", "访问节点", "评估叶子", "剪枝次数"], rows),
        "![](e2_tree_0.png)", "![](e2_tree_1.png)",
    ])


# ---------------------------------------------------------------- E3 状态设计与缓存、规模

METHODS = (
    ("Minimax", lambda g, n: p4.solve(g, n, "minimax")),
    ("AB 自然顺序", lambda g, n: p4.solve(g, n)),
    ("AB 贪心排序", lambda g, n: p4.solve(g, n, order=p4.order_greedy)),
    ("AB + 置换表（键含累计收益）", lambda g, n: p4.solve(g, n, cache=True)),
    ("增量式 Minimax + 缓存", lambda g, n: p4.solve_incremental(g, n, prune=False)),
    ("增量式 AB + 置换表", lambda g, n: p4.solve_incremental(g, n)),
    ("增量式 AB + 置换表 + 贪心", lambda g, n: p4.solve_incremental(g, n, order=p4.order_greedy)),
)
MINIMAX_MAX_N = 8  # 再大 Minimax 太慢（10 个宝藏约 980 万个节点）


def e3_scale(ns, seeds) -> str:
    raw, rows = [], []
    for n in ns:
        acc = {name: ([], [], []) for name, _ in METHODS}
        for seed in seeds:
            grid, values = treasure_map(10, n, seed)
            game = build_game(grid, values)
            ref = p4.solve_incremental(game, n).value
            for name, fn in METHODS:
                if name == "Minimax" and n > MINIMAX_MAX_N:
                    continue
                t0 = time.perf_counter()
                r = fn(game, n)
                ms = (time.perf_counter() - t0) * 1000
                check([r], ref)
                acc[name][0].append(r.stats.nodes)
                acc[name][1].append(r.stats.tt_hits)
                acc[name][2].append(ms)
                raw.append([n, seed, name, r.stats.nodes, r.stats.tt_hits, ms])
        rows.append([n, *[f"{mean(acc[name][0]):.0f}" if acc[name][0] else "-" for name, _ in METHODS]])
        rows.append(["", *[f"命中 {mean(acc[name][1]):.0f}" if acc[name][0] else "" for name, _ in METHODS]])
        rows.append(["", *[f"{mean(acc[name][2]):.1f} ms" if acc[name][0] else "" for name, _ in METHODS]])
    OUTPUT.write_csv("e3_scale.csv", ["n_treasures", "seed", "method", "nodes", "tt_hits", "ms"], raw)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (name, _) in enumerate(METHODS):
        pts = [(n, mean([r[3] for r in raw if r[0] == n and r[2] == name])) for n in ns
               if any(r[0] == n and r[2] == name for r in raw)]
        ax.plot(*zip(*pts), "o-" if "增量" in name else "s--", label=name, color=f"C{i}")
    ax.set(yscale="log", xlabel="宝藏数 N（搜索到终局，H = N）", ylabel="平均访问节点数（对数坐标）",
           title=f"随机 10x10 地图，每档 {len(seeds)} 张")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("e3_scale.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        f"## E3 状态设计、缓存与规模（随机 10x10 地图，每档 {len(seeds)} 张，搜索到终局）",
        "![](e3_scale.png)",
        "每个宝藏数对应三行：平均访问节点数、平均置换表命中次数、平均耗时。",
        viz.md_table(["宝藏数", *[name for name, _ in METHODS]], rows),
    ])


# ---------------------------------------------------------------- E4 视野效应

def e4_horizon(batches: list[list[int]], n=6) -> str:
    """batches：若干批随机地图的种子。同一实验在不同批次上重复，用来区分"规律"和"碰巧"。"""
    _, _, game = load_sample()
    sample_rows = []
    for da in (1, 2, 3, 4):
        s, moves = p4.play(game, da, None)
        sample_rows.append([f"A 看 {da} 层，B 看到终局", " ".join(moves), s.score_a, s.score_b, s.utility])
    for d in (1, 2, 3, 4):
        s, moves = p4.play(game, d, d)
        sample_rows.append([f"双方都看 {d} 层", " ".join(moves), s.score_a, s.score_b, s.utility])

    hs = list(range(1, n + 1))
    agree = {h: [] for h in hs}
    loss = {h: [] for h in hs}               # 全部地图，按地图顺序排列（配对检验要求同一顺序）
    batch_loss = [{h: [] for h in hs} for _ in batches]
    raw = []
    for b, seeds in enumerate(batches):
        for seed in seeds:
            grid, values = treasure_map(10, n, seed)
            game_r = build_game(grid, values)
            full = p4.solve_incremental(game_r, n)
            full_u = p4.play(game_r, None, None)[0].utility
            for h in hs:
                ok = p4.solve_incremental(game_r, h).best_action == full.best_action
                lost = full_u - p4.play(game_r, h, None)[0].utility
                agree[h].append(ok)
                loss[h].append(lost)
                batch_loss[b][h].append(lost)
                raw.append([b + 1, seed, h, int(ok), lost])
    OUTPUT.write_csv("e4_horizon.csv", ["batch", "seed", "depth", "first_move_agrees", "loss"], raw)
    total = sum(len(s) for s in batches)

    se = {h: paired_test([0] * len(loss[h]), loss[h])["se"] for h in hs}  # 均值的标准误
    batch_rows = [[f"第 {b + 1} 批（{len(seeds)} 张）", *[f"{mean(batch_loss[b][h]):.2f}" for h in hs]]
                  for b, seeds in enumerate(batches)]
    batch_rows.append([f"合计（{total} 张）", *[f"{mean(loss[h]):.2f} ± {se[h]:.2f}" for h in hs]])
    batch_rows.append(["首步与全深度一致", *[f"{mean(agree[h]):.0%}" for h in hs]])

    test_rows = []
    for h1, h2 in zip(hs, hs[1:]):
        t = paired_test(loss[h1], loss[h2])
        # 各批次里，差值方向与合计结果相同的批数：显著的规律应当每批都朝同一方向
        same = sum((mean(bl[h2]) - mean(bl[h1])) * t["diff"] > 0 for bl in batch_loss)
        test_rows.append([f"H={h1} → {h2}", f"{t['diff']:+.2f} ± {t['se']:.2f}", f"{t['t']:.2f}",
                          verdict(t), f"{same}/{len(batches)}"])

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for b, bl in enumerate(batch_loss):
        ax.plot(hs, [mean(bl[h]) for h in hs], "-", color="0.7", lw=1,
                label="各批次" if b == 0 else None)
    ax.errorbar(hs, [mean(loss[h]) for h in hs], yerr=[2 * se[h] for h in hs], fmt="o-", color="C3",
                capsize=4, label="合计平均（误差线 = ±2 倍标准误）")
    ax.set(xlabel="A 的搜索深度 H（每回合重新搜索）", ylabel="相对全深度的最终 U 损失")
    ax2 = ax.twinx()
    ax2.plot(hs, [mean(agree[h]) for h in hs], "s--", color="C0", label="首步与全深度一致的比例（右轴）")
    ax2.set_ylim(0, 1.05)
    l1, t1 = ax.get_legend_handles_labels()
    l2, t2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, t1 + t2, fontsize=8, loc="upper right")
    ax.set_title(f"随机地图 {n} 个宝藏，{len(batches)} 批共 {total} 局；B 始终看到终局")
    fig.tight_layout()
    fig.savefig(out("e4_horizon.png"), dpi=150)
    plt.close(fig)

    return "\n\n".join([
        "## E4 视野效应",
        "### 样例：按不同深度实际走完整局（每回合重新搜索）",
        viz.md_table(["设置", "实际走法", "A 总收益", "B 总收益", "最终 U"], sample_rows),
        f"### 随机地图（{n} 个宝藏，{len(batches)} 批共 {total} 张；B 始终看到终局）",
        "![](e4_horizon.png)",
        "表中数字为平均损失 = 全深度对局的最终 U − A 用深度 H 时的最终 U，越小越好；± 为标准误。",
        viz.md_table(["", *[f"H={h}" for h in hs]], batch_rows),
        "### 相邻深度的配对检验（全部地图；差值 = 后者 − 前者，负数表示损失减少，|t| ≥ 2 视为显著）",
        viz.md_table(["比较", "损失差值", "t", "判断", "各批次方向一致"], test_rows),
    ])


# ---------------------------------------------------------------- E5 剪枝空间与最少叶子数

def e5_min_leaves(seeds) -> str:
    """分支数逐层递减（N, N-1, ..., 1）的博弈树能剪多少，由上面的公式决定。"""
    _, _, game = load_sample()
    rows = []
    for h in (2, 3, 4):
        br = [4 - i for i in range(h)]
        mm, nat, gr, ideal = (p4.solve(game, h, "minimax"), p4.solve(game, h), p4.solve(game, h, order=p4.order_greedy),
                              p4.solve(game, h, order=p4.order_oracle(game, h)))
        rows.append([f"样例 H={h}", "x".join(map(str, br)), mm.stats.leaves, nat.stats.leaves, gr.stats.leaves,
                     ideal.stats.leaves, min_leaves(br), f"{ideal.stats.leaves / mm.stats.leaves:.0%}"])
    for n in (5, 6, 7):
        acc = ([], [], [], [])
        for seed in seeds:
            grid, values = treasure_map(10, n, seed)
            g = build_game(grid, values)
            for i, r in enumerate((p4.solve(g, n, "minimax"), p4.solve(g, n), p4.solve(g, n, order=p4.order_greedy),
                                   p4.solve(g, n, order=p4.order_oracle(g, n)))):
                acc[i].append(r.stats.leaves)
        br = [n - i for i in range(n)]
        rows.append([f"随机地图 N={n}", "x".join(map(str, br)), *[mean(a) for a in acc], min_leaves(br),
                     f"{mean(acc[3]) / mean(acc[0]):.0%}"])
    return "\n\n".join([
        "## E5 剪枝空间：完美排序下的最少叶子数（随机地图 U 可能出现并列，理想顺序可略低于公式）",
        viz.md_table(["设置", "各层分支数", "Minimax 叶子", "自然顺序", "贪心顺序", "理想顺序", "公式下界", "理想/Minimax"], rows),
    ])


# ---------------------------------------------------------------- E6 对局损失分解、样例主变例对比

def e6_loss_decomposition(seeds, n=6) -> str:
    """A 共决策 n/2 次，决策时剩余 n, n-2, ... 个宝藏；剩余数 <= H 的决策已能看到终局，必为最优。
    把总损失拆成"第一步决策造成的损失"（第一步之后双方都按最优走）与其余决策造成的损失。"""
    hs = list(range(1, n + 1))
    total, first, wrong = {h: [] for h in hs}, {h: [] for h in hs}, {h: [] for h in hs}
    for seed in seeds:
        grid, values = treasure_map(10, n, seed)
        g = build_game(grid, values)
        s0 = g.initial_state()
        v = p4.solve_incremental(g, n).value
        for h in hs:
            t = p4.solve_incremental(g, h).best_action
            first[h].append(v - p4.solve_incremental(g, n - 1, state=g.result(s0, t)).value)
            total[h].append(v - p4.play(g, h, None)[0].utility)
            wrong[h].append(first[h][-1] > 0)
    rows = [[h, sum(1 for rem in range(n, 0, -2) if rem > h), f"{mean(total[h]):.2f}", f"{mean(first[h]):.2f}",
             f"{mean(total[h]) - mean(first[h]):.2f}", f"{mean(wrong[h]):.0%}"] for h in hs]

    _, _, game = load_sample()
    s0 = game.initial_state()
    lines = []
    for t in ("T1", "T2", "T3", "T4"):
        child = game.result(s0, t)
        r = p4.solve_incremental(game, 3, state=child)
        s = child
        for a in r.pv:
            s = game.result(s, a)
        lines.append([t, p4.explain_pv(game, [t] + r.pv), s.score_a, s.score_b, s.utility])
    replies = []
    child = game.result(s0, "T3")
    for t in sorted(child.remaining, key=p4.point_key):
        c2 = game.result(child, t)
        replies.append([f"B→{t}", game.gain(child, t), p4.solve_incremental(game, 2, state=c2).value])
    return "\n\n".join([
        f"## E6 对局损失分解（{n} 个宝藏，{len(seeds)} 张；B 看到终局）",
        viz.md_table(["A 的深度 H", "看不到终局的决策数", "总损失", "第一步造成的损失", "其余决策造成的损失", "第一步选错的比例"], rows),
        "## E7 样例：H=4 时四个首步各自的主变例",
        viz.md_table(["首步", "主变例", "A 总收益", "B 总收益", "U"], lines),
        "## E8 样例：A 先取 T3 后 B 的选择（H=4）",
        viz.md_table(["B 的选择", "B 的即时收益", "继续搜索后的 U"], replies),
    ])


# ---------------------------------------------------------------- 入口

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="缩小规模，快速检查脚本能否跑通")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.prepare(args.quick)
    if args.quick:
        ns, seeds, batches = (4, 5, 6), list(range(2)), [[0, 1, 2], [3, 4, 5]]
    else:
        # E4 用 4 批、每批 50 张不同的地图重复，检验结论在不同样本上是否稳定
        ns, seeds, batches = tuple(range(4, 11)), list(range(10)), \
            [list(range(i * 50, (i + 1) * 50)) for i in range(4)]

    sections = ["# 题目四 对照实验结果", "由 `python -m experiments.exp_p4` 自动生成。"]
    for name, fn in (("E1", e1_sample), ("E2", e2_orders),
                     ("E3", lambda: e3_scale(ns, seeds)),
                     ("E4", lambda: e4_horizon(batches)),
                     ("E5", lambda: e5_min_leaves(seeds)),
                     ("E6", lambda: e6_loss_decomposition([s for b in batches for s in b]))):
        print(f"运行 {name} ...", flush=True)
        sections.append(fn())
    print(f"完成，结果在 {OUTPUT.write_summary(sections)}")


if __name__ == "__main__":
    main()
