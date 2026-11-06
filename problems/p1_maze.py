"""题目一：迷宫寻路。

'X' 不可通行，其余格子（'.'、'S'、'E'）均可通行；四方向移动，每步代价 1。
输出最短路径的坐标序列，无解返回 []。
"""
from __future__ import annotations

from core.astar import SearchResult, astar
from core.grid import (Coord, h_euclidean, h_manhattan, h_zero, in_bounds,
                       load_grid_problem, make_neighbors)

# 均为可采纳的启发函数；weight > 1 时即加权 A*，不再保证最优
HEURISTICS = ("manhattan", "zero", "euclidean")


def cell_cost(token: str):
    return None if token == "X" else 1


def solve(start: Coord, goal: Coord, grid: list[list[str]], heuristic: str = "manhattan",
          tie_break: str = "low_h", trace: bool = False, weight: float = 1.0) -> SearchResult:
    """返回完整的搜索结果（路径 + 统计），供实验对比使用。"""
    for p in (start, goal):
        if not in_bounds(grid, p) or cell_cost(grid[p[0]][p[1]]) is None:
            return SearchResult()  # 起点或终点非法，视为无解
    h = {"manhattan": h_manhattan, "zero": h_zero, "euclidean": h_euclidean}[heuristic](goal)
    if weight != 1.0:
        base = h
        h = lambda n: weight * base(n)
    return astar(start, goal, make_neighbors(grid, cell_cost), h, tie_break, trace)


def find_path(start: Coord, goal: Coord, grid: list[list[str]]) -> list[Coord]:
    """题目要求的接口：返回 List[Tuple[int, int]]，无解返回 []。"""
    return solve(start, goal, grid).path


def render(grid: list[list[str]], path: list[Coord]) -> str:
    """把路径画回地图上，便于肉眼核对（路径上的格子标成 *）。"""
    canvas = [row[:] for row in grid]
    for r, c in path[1:-1]:
        canvas[r][c] = "*"
    return "\n".join(" ".join(row) for row in canvas)


def run(path: str) -> None:
    start, goal, grid = load_grid_problem(path)
    res = solve(start, goal, grid)
    print(res.path)
    if res.found:
        print(f"步数 {len(res.path) - 1}，扩展节点 {res.expanded}，入队 {res.generated}")
        print(render(grid, res.path))
    else:
        print(f"无可行路径，扩展节点 {res.expanded}")
