"""题目二：加权地形。

数字表示进入该格的代价；进入终点 E 的代价为 0（题目说明"最后一步没有代价"）。
'S' 的代价记为 1：路径代价全为正，最优路径不会回到起点，所以这个取值不影响结果。
'X' 视为障碍（样例中没有出现，额外支持）。
输出最小总代价，无解返回 -1。

启发函数里"每步至少花多少"（最小进入代价 w_min）统计时排除起点和终点：
  - 终点的进入代价单独处理（见 h_goal_aware）；
  - 从起点出发的最优路径不会再进入起点，起点的代价不会出现在任何"剩余代价"里。
    若把它计入，S 记的 1 会把 w_min 拉低：地图上最便宜的地形是 3 时，启发函数会按 1 估计，几乎失效。
"""
from __future__ import annotations

from core.astar import SearchResult, astar
from core.grid import (Coord, h_goal_aware, h_manhattan, h_zero, in_bounds,
                       load_grid_problem, make_neighbors, min_step_cost)

# goal_aware 为默认：可采纳且一致；manhattan 在本题会高估 1，留作对照实验
HEURISTICS = ("goal_aware", "manhattan", "zero")
# 仅用于对照实验：
#   manhattan_avg         ：系数取平均代价，是常见的错误做法，不可采纳，可能得到次优解
#   goal_aware_incl_start ：改进前的版本，统计 w_min 时把起点也算进去
EXPERIMENTAL_HEURISTICS = ("manhattan_avg", "goal_aware_incl_start")


def cell_cost(token: str):
    if token == "X":
        return None
    if token == "E":
        return 0
    if token == "S":
        return 1
    w = float(token)
    if w < 0:
        raise ValueError(f"A* 要求代价非负，遇到 {token}")
    return int(w) if w.is_integer() else w


def make_heuristic(name: str, goal: Coord, grid: list[list[str]], start: Coord | None = None):
    """构造启发函数。传入 start 时，统计最小进入代价会把起点排除在外（见模块说明）。"""
    if name == "zero":
        return h_zero(goal)
    # 最小进入代价，作为"每步至少花多少"的下界
    exclude = (goal,) if start is None or name == "goal_aware_incl_start" else (goal, start)
    w_min = min_step_cost(grid, cell_cost, exclude=exclude)
    if name == "manhattan":
        return h_manhattan(goal, scale=w_min)
    if name in ("goal_aware", "goal_aware_incl_start"):
        return h_goal_aware(goal, w_min, cell_cost(grid[goal[0]][goal[1]]))
    if name == "manhattan_avg":
        costs = [cell_cost(t) for r, row in enumerate(grid) for c, t in enumerate(row)
                 if (r, c) != goal and cell_cost(t) is not None]
        return h_manhattan(goal, scale=sum(costs) / len(costs))
    raise ValueError(f"未知启发函数: {name}")


def solve(start: Coord, goal: Coord, grid: list[list[str]], heuristic: str = "goal_aware",
          tie_break: str = "low_h", trace: bool = False) -> SearchResult:
    for p in (start, goal):
        if not in_bounds(grid, p) or cell_cost(grid[p[0]][p[1]]) is None:
            return SearchResult()
    h = make_heuristic(heuristic, goal, grid, start)
    return astar(start, goal, make_neighbors(grid, cell_cost), h, tie_break, trace)


def min_cost(start: Coord, goal: Coord, grid: list[list[str]]) -> int:
    """题目要求的接口：返回最小总代价，无解返回 -1。"""
    res = solve(start, goal, grid)
    return res.cost if res.found else -1


def cost_breakdown(grid: list[list[str]], path: list[Coord]) -> list[tuple[Coord, float, float]]:
    """路径代价的累计过程：[(坐标, 本步代价, 累计代价), ...]，起点本步代价为 0。"""
    out, total = [], 0
    for i, (r, c) in enumerate(path):
        step = 0 if i == 0 else cell_cost(grid[r][c])
        total += step
        out.append(((r, c), step, total))
    return out


def run(path: str) -> None:
    start, goal, grid = load_grid_problem(path)
    res = solve(start, goal, grid)
    print(res.cost if res.found else -1)
    if res.found:
        print("累计过程: " + " -> ".join(f"{p}+{w}={t}" for p, w, t in cost_breakdown(grid, res.path)))
        print(f"扩展节点 {res.expanded}，入队 {res.generated}，重新打开 {res.reopened}")
