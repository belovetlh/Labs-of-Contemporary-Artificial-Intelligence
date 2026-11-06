"""实验用地图生成器。

所有随机地图都由 seed 唯一确定：同一组参数永远生成同一张图，保证实验可复现。
返回值统一为 (grid, start, goal)，grid 中起点标 'S'、终点标 'E'、障碍标 'X'、空地标 '.'。
"""
from __future__ import annotations

import random

from core.astar import astar
from core.grid import Coord, h_manhattan, make_neighbors

MAX_TRIES = 1000


def _passable(token: str):
    return None if token == "X" else 1


def is_solvable(grid: list[list[str]], start: Coord, goal: Coord) -> bool:
    return astar(start, goal, make_neighbors(grid, _passable), h_manhattan(goal)).found


def _mark(grid: list[list[str]], start: Coord, goal: Coord) -> list[list[str]]:
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"
    return grid


def random_grid(size: int, density: float, seed: int, endpoints: str = "corner"):
    """size x size 的随机障碍地图。

    endpoints="corner"：起点在左上角、终点在右下角（终点离起点最远，对 h=0 最不利）；
    endpoints="random"：起点、终点从可通行格中随机选取。
    每个格子以概率 density 设为障碍。无解时用同一个随机数生成器继续重新生成，
    因此"第 seed 张图"仍是确定的。
    """
    rng = random.Random(seed if endpoints == "corner" else f"random-endpoints-{seed}")
    for _ in range(MAX_TRIES):
        grid = [["X" if rng.random() < density else "." for _ in range(size)]
                for _ in range(size)]
        if endpoints == "corner":
            start, goal = (0, 0), (size - 1, size - 1)
        else:
            free = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == "."]
            start, goal = rng.sample(free, 2)
        _mark(grid, start, goal)
        if is_solvable(grid, start, goal):
            return grid, start, goal
    raise RuntimeError(f"size={size}, density={density} 下 {MAX_TRIES} 次都没有生成出有解地图")


def weighted_grid(size: int, seed: int, p_high: float = 0.3, high: tuple = (3, 8),
                  low: int = 1, density: float = 0.0):
    """题目二用的加权地形，起点左上、终点右下，格子为代价数字。

    每格：以概率 density 设为障碍 X；否则以概率 p_high 从 high 中随机取一个高代价（沼泽等）；
    其余为 low（平原）。默认 low=1、high=(3, 8)，与题目样例中的 1/3/8 一致。
    """
    rng = random.Random(seed)
    start, goal = (0, 0), (size - 1, size - 1)
    for _ in range(MAX_TRIES):
        grid = [["X" if rng.random() < density else
                 str(rng.choice(high)) if rng.random() < p_high else str(low)
                 for _ in range(size)] for _ in range(size)]
        _mark(grid, start, goal)
        if is_solvable(grid, start, goal):
            return grid, start, goal
    raise RuntimeError(f"size={size}, density={density} 下 {MAX_TRIES} 次都没有生成出有解地图")


def treasure_map(size: int, n_treasures: int, seed: int, p_high: float = 0.3,
                 density: float = 0.15, value_range: tuple = (5, 12)):
    """题目四用的随机地图：加权地形 + 障碍，随机放置 A、B 与 T1..Tn，宝藏价值随机。

    返回 (grid, values)。保证所有关键点两两连通（否则重新生成）。
    """
    rng = random.Random(f"treasure-{seed}")
    names = ["A", "B"] + [f"T{i}" for i in range(1, n_treasures + 1)]
    for _ in range(MAX_TRIES):
        grid = [["X" if rng.random() < density else
                 str(rng.choice((3, 8))) if rng.random() < p_high else "1"
                 for _ in range(size)] for _ in range(size)]
        free = [(r, c) for r in range(size) for c in range(size) if grid[r][c] != "X"]
        if len(free) < len(names):
            continue
        cells = rng.sample(free, len(names))
        for name, (r, c) in zip(names, cells):
            grid[r][c] = name
        if all(is_solvable(grid, cells[0], p) for p in cells[1:]):
            values = {n: rng.randint(*value_range) for n in names[2:]}
            return grid, values
    raise RuntimeError("没有生成出连通的宝藏地图")


def open_grid(size: int):
    """没有任何障碍：曼哈顿距离恰好等于真实距离，是启发函数表现最好的情形。"""
    grid = [["."] * size for _ in range(size)]
    start, goal = (size // 2, 1), (size // 2, size - 2)
    return _mark(grid, start, goal), start, goal


def u_trap_grid(size: int):
    """U 形陷阱：起点在一个开口朝左的 U 形墙内部，终点在墙的右后方。

        . . . . . . . .
        . X X X X X . .
        . . . . . X . .
        . . S . . X . E      <- 曼哈顿距离把搜索一直往右拉，撞到墙底
        . . . . . X . .
        . X X X X X . .
        . . . . . . . .

    要到达终点，必须先往左退出 U 形口，再绕过上沿或下沿。
    曼哈顿距离会把 U 内部几乎所有格子都展开完，才肯"回头"。
    """
    grid = [["."] * size for _ in range(size)]
    top, bottom = size // 4, size - 1 - size // 4
    left, wall_col = size // 6, size * 2 // 3
    for c in range(left, wall_col + 1):
        grid[top][c] = "X"
        grid[bottom][c] = "X"
    for r in range(top, bottom + 1):
        grid[r][wall_col] = "X"
    mid = size // 2
    start, goal = (mid, (left + wall_col) // 2), (mid, size - 2)
    return _mark(grid, start, goal), start, goal
