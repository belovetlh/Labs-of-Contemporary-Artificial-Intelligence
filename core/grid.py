"""二维网格：输入解析、邻居生成、启发函数。

坐标统一为 (row, col)，只允许上下左右四个方向移动。
"""
from __future__ import annotations

import re
from typing import Callable, Optional

Coord = tuple[int, int]
# 返回进入该格的代价；None 表示不可通行
CostFn = Callable[[str], Optional[float]]

# 方向顺序固定为 上、下、左、右。存在多条等长最短路时，它决定返回哪一条
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_COORD_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_ROW_RE = re.compile(r"\[([^\[\]]*)\]")  # 最内层的 [...]，即地图的一行


# ---------------------------------------------------------------- 解析

def parse_grid(text: str) -> list[list[str]]:
    """解析 [[...], [...]] 形式的地图。

    元素带不带引号都可以：题目一的样例是 'S'、'.'，题目二的样例是裸的 S、1、E。
    """
    grid = []
    for row_text in _ROW_RE.findall(text):
        row = [tok.strip().strip("'\"") for tok in row_text.split(",")]
        row = [tok for tok in row if tok]  # 容忍行尾多余的逗号
        if row:
            grid.append(row)
    if not grid:
        raise ValueError("没有解析到地图")
    if len({len(r) for r in grid}) != 1:
        raise ValueError("地图各行长度不一致")
    return grid


def parse_grid_problem(text: str) -> tuple[Coord, Coord, list[list[str]]]:
    """解析 题目一/二 的输入：第一个坐标是起点，第二个是终点，其后是地图。"""
    matches = list(_COORD_RE.finditer(text))
    if len(matches) < 2:
        raise ValueError("输入中需要先给出起点和终点坐标")
    start = (int(matches[0].group(1)), int(matches[0].group(2)))
    goal = (int(matches[1].group(1)), int(matches[1].group(2)))
    grid = parse_grid(text[matches[1].end():])
    return start, goal, grid


def load_grid_problem(path: str) -> tuple[Coord, Coord, list[list[str]]]:
    with open(path, encoding="utf-8-sig") as f:
        return parse_grid_problem(f.read())


# ---------------------------------------------------------------- 邻居

def in_bounds(grid: list[list[str]], pos: Coord) -> bool:
    return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])


def make_neighbors(grid: list[list[str]], cost_fn: CostFn):
    """生成 A* 所需的 neighbors 函数：一步的代价 = 进入目标格的代价。"""
    def neighbors(node: Coord):
        r, c = node
        for dr, dc in DIRECTIONS:
            nxt = (r + dr, c + dc)
            if in_bounds(grid, nxt):
                w = cost_fn(grid[nxt[0]][nxt[1]])
                if w is not None:
                    yield nxt, w
    return neighbors


def min_step_cost(grid: list[list[str]], cost_fn: CostFn, exclude: tuple = ()) -> float:
    """地图上所有可通行格（exclude 中的坐标除外）的最小进入代价。"""
    costs = [cost_fn(tok)
             for r, row in enumerate(grid) for c, tok in enumerate(row)
             if (r, c) not in exclude]
    costs = [w for w in costs if w is not None]
    return min(costs) if costs else 0


# ---------------------------------------------------------------- 启发函数

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_zero(goal: Coord):
    """h = 0：A* 退化为 Dijkstra / 一致代价搜索，作为对照基线。"""
    return lambda n: 0


def h_manhattan(goal: Coord, scale: float = 1):
    """h = scale * 曼哈顿距离。

    每步代价都不低于 scale 时可采纳：至少要走 manhattan 步，每步至少花 scale。
    """
    return lambda n: scale * manhattan(n, goal)


def h_euclidean(goal: Coord, scale: float = 1):
    """h = scale * 欧氏距离。

    欧氏距离 <= 曼哈顿距离，因此同样可采纳，但在四方向网格上更"松"（离真实代价更远），
    用来说明：同为可采纳时，h 越接近真实代价，展开的节点越少。
    """
    return lambda n: scale * ((n[0] - goal[0]) ** 2 + (n[1] - goal[1]) ** 2) ** 0.5


def h_goal_aware(goal: Coord, w_min: float, w_goal: float):
    """考虑终点进入代价的曼哈顿启发。

    从 n 出发至少走 d = manhattan(n, goal) 步：最后一步进入终点，花 w_goal；
    前 d-1 步各至少花 w_min。所以 h(n) = (d-1)*w_min + w_goal，h(goal) = 0。
    题目二中 w_goal = 0（进入 E 无代价），普通曼哈顿会高估 1，而这个版本可采纳且一致。
    """
    def h(n: Coord) -> float:
        d = manhattan(n, goal)
        return 0 if d == 0 else (d - 1) * w_min + w_goal
    return h
