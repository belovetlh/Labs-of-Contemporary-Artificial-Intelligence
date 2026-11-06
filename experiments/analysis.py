"""分析实验用到的纯计算函数：不依赖画图库，也不写文件，可以直接被单元测试调用。"""
from __future__ import annotations

from collections import deque

from core.grid import Coord, make_neighbors, manhattan


def distances_from(grid, start: Coord, cost_fn) -> dict[Coord, int]:
    """等权网格上 BFS 求起点到各格的最短步数 g*(n)。"""
    neighbors = make_neighbors(grid, cost_fn)
    dist, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for nxt, _ in neighbors(node):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist


def classify_by_f(grid, start: Coord, goal: Coord, cost_fn) -> dict:
    """按 f = g* + 曼哈顿距离 与最优代价 C* 的关系给格子分类。

    启发函数一致时：f < C* 的节点一定展开，f > C* 的一定不展开，f = C* 的取决于平局规则。
    """
    gstar = distances_from(grid, start, cost_fn)
    cstar = gstar[goal]
    f = {n: d + manhattan(n, goal) for n, d in gstar.items()}
    return {"cstar": cstar, "gap": cstar - manhattan(start, goal),
            "must": {n for n, v in f.items() if v < cstar},
            "tied": {n for n, v in f.items() if v == cstar}}


def cheap_connected(grid, start: Coord, goal: Coord, monotone: bool) -> bool:
    """起点能否只经过平原（代价 1）格到达终点。monotone=True 时只允许向下、向右（不绕路）。"""
    def cheap(p):
        return 0 <= p[0] < len(grid) and 0 <= p[1] < len(grid[0]) and grid[p[0]][p[1]] in ("1", "E")
    steps = ((1, 0), (0, 1)) if monotone else ((1, 0), (0, 1), (-1, 0), (0, -1))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        if (r, c) == goal:
            return True
        for dr, dc in steps:
            nxt = (r + dr, c + dc)
            if nxt not in seen and cheap(nxt):
                seen.add(nxt)
                stack.append(nxt)
    return False


def min_leaves(branching: list[int]) -> int:
    """广义 Knuth-Moore：各层分支数为 b1, b2, ... 时，完美排序下需评估的叶子数
    = (奇数层分支数之积) + (偶数层分支数之积) - 1。"""
    odd, even = 1, 1
    for i, b in enumerate(branching):
        if i % 2 == 0:
            odd *= b
        else:
            even *= b
    return odd + even - 1
