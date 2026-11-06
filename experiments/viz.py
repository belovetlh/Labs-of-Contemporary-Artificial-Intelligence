"""搜索过程可视化：展开区域图、逐步表格。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接输出到文件
import matplotlib.pyplot as plt
import numpy as np

from core.astar import SearchResult

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FREE = (0.96, 0.96, 0.96)
WALL = (0.18, 0.18, 0.18)


def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _terrain(grid) -> np.ndarray:
    """底图：障碍为深色；数字格按代价着灰度（越贵越深）；其他格为浅色。"""
    nums = [int(t) for row in grid for t in row if t.isdigit()]
    lo, hi = (min(nums), max(nums)) if nums else (0, 0)

    def color(t):
        if t == "X":
            return WALL
        if t.isdigit() and hi > lo:
            v = 0.96 - 0.45 * (int(t) - lo) / (hi - lo)
            return (v, v, v)
        return FREE
    return np.array([[color(t) for t in row] for row in grid])


def draw_expansion(ax, grid, res: SearchResult, title: str, vmax: int,
                   start=None, goal=None, show_order: bool = False):
    """在 ax 上画一张图：深色=障碍，彩色=被展开过的格子（颜色越亮越晚被展开），红线=最终路径。"""
    rows, cols = len(grid), len(grid[0])
    ax.imshow(_terrain(grid), interpolation="nearest")

    order = np.full((rows, cols), np.nan)
    for i, (r, c) in enumerate(res.expanded_order, 1):
        order[r, c] = i
    im = ax.imshow(np.ma.masked_invalid(order), cmap="viridis", vmin=1, vmax=max(vmax, 2),
                   interpolation="nearest", alpha=0.9)

    if show_order:
        for (r, c), v in np.ndenumerate(order):
            if not np.isnan(v):
                ax.text(c, r, int(v), ha="center", va="center", fontsize=9,
                        color="white" if v < vmax * 0.6 else "black")
    if res.path:
        ys, xs = zip(*res.path)
        ax.plot(xs, ys, color="red", lw=2.2 if rows <= 30 else 1.4)
    start = start or (res.path[0] if res.path else None)
    goal = goal or (res.path[-1] if res.path else None)
    # 标了展开序号时，把起终点标记挪到格子左上角，避免盖住数字
    off = -0.3 if show_order else 0
    size = 0.6 if show_order else 1
    if start:
        ax.plot(start[1] + off, start[0] + off, "o", color="lime", ms=9 * size, mec="black")
    if goal:
        ax.plot(goal[1] + off, goal[0] + off, "*", color="red", ms=14 * size, mec="black")

    ax.set_xticks([])
    ax.set_yticks([])
    if not res.path:
        length = "，无解"
    elif res.cost != len(res.path) - 1:  # 加权地图：步数与代价不同，两者都标出
        length = f"，路径 {len(res.path) - 1} 步、代价 {res.cost:g}"
    else:
        length = f"，路径长度 {len(res.path) - 1}"
    ax.set_title(f"{title}\n展开 {res.expanded} 个节点{length}", fontsize=10)
    return im


def compare_expansions(grid, runs: list[tuple[str, SearchResult]], out_path: str,
                       suptitle: str = "", show_order: bool = False, start=None, goal=None) -> None:
    """多种搜索设置并排比较。所有子图共用同一色标，颜色可直接横向比较。"""
    vmax = max(res.expanded for _, res in runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.3 * len(runs), 4.8), squeeze=False)
    im = None
    for ax, (title, res) in zip(axes[0], runs):
        im = draw_expansion(ax, grid, res, title, vmax, start, goal, show_order)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.75, label="展开顺序")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_paths(grid, paths: list[tuple[str, list, str]], out_path: str, title: str = "",
               annotate: bool = False) -> None:
    """在地形图上叠加多条路径：paths = [(图例, 路径, 颜色), ...]。annotate=True 时在格子里标出代价。"""
    cols = len(grid[0])
    fig, ax = plt.subplots(figsize=(max(4, cols * 0.55 if annotate else 6),) * 2)
    _draw_paths_on(ax, grid, paths, title, annotate)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_path_panels(grid, panels: list[tuple[str, list]], out_path: str, suptitle: str = "",
                     annotate: bool = True) -> None:
    """多张路径图并排：panels = [(子图标题, paths), ...]。"""
    cols = len(grid[0])
    side = max(4, cols * 0.55)
    fig, axes = plt.subplots(1, len(panels), figsize=(side * len(panels), side + 1), squeeze=False)
    for ax, (title, paths) in zip(axes[0], panels):
        _draw_paths_on(ax, grid, paths, title, annotate)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_paths_on(ax, grid, paths, title, annotate) -> None:
    rows, cols = len(grid), len(grid[0])
    ax.imshow(_terrain(grid), interpolation="nearest")
    if annotate:
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, grid[r][c], ha="center", va="center", fontsize=11,
                        color="white" if grid[r][c] == "X" else "black")
    for i, (label, path, color) in enumerate(paths):
        if path:
            ys, xs = zip(*path)
            shift = (i - (len(paths) - 1) / 2) * 0.12  # 多条路径重合时错开一点，避免互相遮挡
            ax.plot([x + shift for x in xs], [y + shift for y in ys], color=color, lw=2.5,
                    label=label, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9, ncol=1)
    if title:
        ax.set_title(title, fontsize=11)


def step_table_md(res: SearchResult, limit: int | None = None) -> str:
    """逐步表格（Markdown）。每行是一次出队-扩展；最后一行补上终点出队。"""
    lines = ["| 轮次 | 出队节点 | g | h | f | 父节点 | 新加入开放表 (节点:f) | 开放表大小 |",
             "|---|---|---|---|---|---|---|---|"]
    steps = res.steps if limit is None else res.steps[:limit]
    for s in steps:
        pushed = ", ".join(f"{n}:{_fmt(f)}" for n, _, _, f in s.pushed) or "（无）"
        lines.append(f"| {s.step} | {s.node} | {_fmt(s.g)} | {_fmt(s.h)} | {_fmt(s.f)} | "
                     f"{s.parent if s.parent is not None else '-'} | {pushed} | {s.open_size} |")
    if limit is not None and len(res.steps) > limit:
        lines.append(f"| ... | 共 {len(res.steps)} 步，此处省略 | | | | | | |")
    elif res.found:
        lines.append(f"| 终止 | {res.path[-1]} | {_fmt(res.cost)} | 0 | {_fmt(res.cost)} | "
                     f"{res.path[-2] if len(res.path) > 1 else '-'} | 终点出队，搜索结束 | |")
    return "\n".join(lines)


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        out.append("| " + " | ".join(_fmt(round(x, 3) if isinstance(x, float) else x)
                                     for x in row) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- 博弈树

_MAX_COLOR, _MIN_COLOR, _LEAF_COLOR = "#f6d0cc", "#cfe0f3", "#ececec"


def fmt_num(v) -> str:
    """数值格式化：无穷显示为 ±∞，浮点保留 3 位有效数字。"""
    if v == float("inf"):
        return "+∞"
    if v == float("-inf"):
        return "-∞"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def draw_game_tree(ax, root, trace=None, title: str = "", edge_labels: dict | None = None) -> None:
    """画博弈树。

    trace=None：画 Minimax 视角，每个节点标出回传值。
    trace=NodeTrace 列表：画 Alpha-Beta 视角，节点下方标出 #访问序号 和进入时的 [α, β]；
    未被访问（被剪掉）的节点用灰色虚线表示，发生剪枝的那条边用红色虚线表示。
    edge_labels：{子节点路径: 文字}，标在对应的边上（如题目四的走法名称）。
    """
    visited = {t.path: t for t in trace} if trace is not None else None
    pos, items = {}, []
    leaf_x = [0]

    def layout(node, path, depth, is_max):
        flag = getattr(node, "is_max", None)
        is_max = is_max if flag is None else flag
        if not node.children:
            x = leaf_x[0]
            leaf_x[0] += 1
            value = node.value
        else:
            xs, vals = [], []
            for i, c in enumerate(node.children):
                cx, cv = layout(c, path + (i,), depth + 1, not is_max)
                xs.append(cx)
                vals.append(cv)
            x = sum(xs) / len(xs)
            value = max(vals) if is_max else min(vals)
        pos[path] = (x, -depth)
        items.append((path, node, is_max, value))
        return x, value
    layout(root, (), 0, True)

    cut_edges = {p + (i,) for t in (trace or []) if t.cut for p, i in [(t.path, j) for j in t.cut]}
    for path, node, is_max, value in items:
        if path:
            parent = path[:-1]
            seen = visited is None or path in visited
            style = dict(color="black", lw=1.2) if seen else \
                dict(color="red" if path in cut_edges else "0.65", lw=1.2, ls="--")
            (x0, y0), (x1, y1) = pos[parent], pos[path]
            ax.plot([x0, x1], [y0, y1], zorder=1, **style)
            if edge_labels and path in edge_labels:
                # 标在靠近子节点的一端：越往下兄弟节点越挤，靠近子节点处的水平间距最大
                ax.text(x0 + (x1 - x0) * 0.78, y0 + (y1 - y0) * 0.78, edge_labels[path], fontsize=7.5,
                        color="#1f4e8c" if seen else "0.6", ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
            if path in cut_edges:
                ax.text(x0 + (x1 - x0) * 0.55, y0 + (y1 - y0) * 0.55, "剪", color="red", fontsize=9,
                        ha="center", va="center", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", lw=0.8))

    for path, node, is_max, value in items:
        x, y = pos[path]
        leaf = not node.children
        rec = visited.get(path) if visited is not None else None
        seen = visited is None or rec is not None
        shown = fmt_num(rec.value if rec is not None else value)
        if rec is not None and rec.cut:
            # 发生剪枝的节点没看完所有子节点，返回的只是界：MAX 处为下界，MIN 处为上界
            shown = ("≥" if is_max else "≤") + shown
        label = shown if leaf else f"{'MAX' if is_max else 'MIN'}\n{shown}"
        if not seen:
            label = fmt_num(value) if leaf else "MAX" if is_max else "MIN"
        fc = (_LEAF_COLOR if leaf else _MAX_COLOR if is_max else _MIN_COLOR) if seen else "white"
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=3,
                color="black" if seen else "0.6",
                bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="black" if seen else "0.6",
                          ls="-" if seen else "--"))
        if rec is not None and not leaf:
            note = f"#{rec.index} [{fmt_num(rec.alpha)}, {fmt_num(rec.beta)}]"
            ax.text(x, y - 0.3, note, ha="center", va="top", fontsize=7.5, color="0.2", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9))
        elif rec is not None:
            ax.text(x, y - 0.25, f"#{rec.index}", ha="center", va="top", fontsize=7, color="0.35")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.5)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def save_game_trees(root, panels: list[tuple[str, list | None]], out_path: str,
                    suptitle: str = "", ncols: int = 2, panel_size: tuple = (6.2, 3.9),
                    edge_labels: dict | None = None) -> None:
    """多棵博弈树并排：panels = [(子图标题, trace 或 None), ...]。"""
    nrows = -(-len(panels) // ncols)
    ncols = min(ncols, len(panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    for ax, (title, trace) in zip(axes.flat, panels):
        draw_game_tree(ax, root, trace, title, edge_labels)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
