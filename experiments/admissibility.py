"""启发函数性质检查：可采纳性（h <= h*）与一致性（h(n) <= c(n,m) + h(m)）。

h*(n) 为 n 到终点的真实最小代价，通过从终点出发的"反向 Dijkstra"一次性求出全部格子的值：
原图中 n -> m 的代价是 cost(m)（进入 m 的代价），所以反向从 m 松弛到邻居 n 时，边权取 cost(m)。

blocked：计算 h* 时视为不可通行的格子。从起点 S 出发的搜索里，剩余路径不会再经过 S，
所以检查"对这次搜索是否可采纳"时，应传 blocked=(S,)，只与不经过 S 的路径比较。
"""
from __future__ import annotations

import heapq

from core.grid import DIRECTIONS, Coord, in_bounds, make_neighbors

EPS = 1e-9


def true_cost_to_goal(grid, goal: Coord, cost_fn, blocked: tuple = ()) -> dict[Coord, float]:
    """返回 {格子: h*(格子)}，只包含能到达终点的格子。"""
    dist = {goal: 0}
    pq = [(0, goal)]
    while pq:
        d, m = heapq.heappop(pq)
        if d > dist[m]:
            continue
        wm = cost_fn(grid[m[0]][m[1]])  # 从任一邻居 n 走进 m 要花的代价
        for dr, dc in DIRECTIONS:
            n = (m[0] + dr, m[1] + dc)
            if in_bounds(grid, n) and n not in blocked and cost_fn(grid[n[0]][n[1]]) is not None:
                nd = d + wm
                if nd < dist.get(n, float("inf")):
                    dist[n] = nd
                    heapq.heappush(pq, (nd, n))
    return dist


def audit(grid, goal: Coord, h, cost_fn, hstar: dict | None = None, blocked: tuple = ()) -> dict:
    """统计启发函数 h 在这张地图上违反可采纳性、一致性的情况（不考虑经过 blocked 的路径）。"""
    hstar = hstar if hstar is not None else true_cost_to_goal(grid, goal, cost_fn, blocked)
    over = [h(n) - v for n, v in hstar.items() if h(n) > v + EPS]
    neighbors = make_neighbors(grid, cost_fn)
    edges = bad_edges = 0
    for n in hstar:
        for m, w in neighbors(n):
            if m in blocked:
                continue
            edges += 1
            if h(n) > w + h(m) + EPS:
                bad_edges += 1
    return {
        "nodes": len(hstar),
        "over_nodes": len(over),                  # h > h* 的格子数
        "over_max": max(over, default=0),         # 最大高估量
        "edges": edges,
        "inconsistent_edges": bad_edges,          # 违反一致性的有向边数
    }
