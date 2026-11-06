"""通用 A* 搜索。

与具体问题解耦，调用方只需提供：
  - neighbors(node) -> 可迭代的 (邻居节点, 一步代价)
  - heuristic(node) -> h(n)，即 node 到终点剩余代价的估计
本模块负责 g/h/f 的维护、优先队列、父指针回溯以及搜索统计。

题目一、二、四都复用这一份实现，区别只在于 neighbors（代价规则）和 heuristic。
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Optional

Node = Hashable
NeighborFn = Callable[[Node], Iterable[tuple[Node, float]]]
HeuristicFn = Callable[[Node], float]

INF = float("inf")

# f 值相同时的出队顺序：
#   low_h : 优先 h 更小（更靠近终点）的节点，通常能显著减少扩展数
#   fifo  : 先入队先出队
#   lifo  : 后入队先出队
TIE_BREAKS = ("low_h", "fifo", "lifo")


@dataclass
class TraceStep:
    """一次出队-扩展的完整记录，用于逐步表格和展开区域图。"""
    step: int             # 第几次扩展（从 1 开始）
    node: Node
    g: float
    h: float
    f: float
    parent: Optional[Node]
    pushed: list          # 本步新加入/更新到开放表的邻居：[(节点, g, h, f), ...]
    open_size: int        # 本步结束时开放表的条目数（含懒惰删除留下的过期条目）


@dataclass
class SearchResult:
    path: list = field(default_factory=list)  # 起点到终点的节点序列；无解为 []
    cost: Optional[float] = None              # 路径总代价；无解为 None
    expanded: int = 0     # 出队并扩展的节点数，衡量搜索量的主要指标
    generated: int = 0    # 入队次数（含因发现更优 g 而重复入队）
    reopened: int = 0     # 已关闭节点被更优路径重新打开的次数；启发函数一致时恒为 0
    max_open: int = 0     # 开放表峰值大小，反映内存开销
    elapsed: float = 0.0  # 耗时（秒）
    steps: list = field(default_factory=list)  # trace=True 时记录每一步的 TraceStep

    @property
    def found(self) -> bool:
        return bool(self.path)

    @property
    def expanded_order(self) -> list:
        return [s.node for s in self.steps]


def astar(start: Node, goal: Node, neighbors: NeighborFn, heuristic: HeuristicFn,
          tie_break: str = "low_h", trace: bool = False) -> SearchResult:
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"未知的 tie_break: {tie_break}")
    t0 = time.perf_counter()
    seq = itertools.count()

    def key(g: float, h: float) -> tuple:
        # 序号 s 全局唯一，保证堆中元组比较不会落到节点本身上
        s = next(seq)
        if tie_break == "low_h":
            return (g + h, h, s)
        if tie_break == "fifo":
            return (g + h, s)
        return (g + h, -s)

    res = SearchResult(generated=1, max_open=1)
    open_heap = [(key(0, heuristic(start)), start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in closed:
            # 懒惰删除：同一节点可能以不同 g 多次入队，只处理第一次（g 最小的那次）
            continue
        if node == goal:
            # 出队时才做终点检测：此时 g(goal) 已是最优；入队时检测在加权图上可能得到次优解
            res.path = _reconstruct(parent, goal)
            res.cost = g[goal]
            break
        closed.add(node)
        res.expanded += 1
        pushed = []

        for nxt, w in neighbors(node):
            ng = g[node] + w
            if ng < g.get(nxt, INF):
                if nxt in closed:
                    # 启发函数不一致时，已关闭节点可能被找到更短路径，需要重新打开以保证最优
                    closed.remove(nxt)
                    res.reopened += 1
                g[nxt] = ng
                parent[nxt] = node
                hn = heuristic(nxt)
                heapq.heappush(open_heap, (key(ng, hn), nxt))
                res.generated += 1
                if trace:
                    pushed.append((nxt, ng, hn, ng + hn))
        res.max_open = max(res.max_open, len(open_heap))
        if trace:
            h = heuristic(node)
            res.steps.append(TraceStep(res.expanded, node, g[node], h, g[node] + h,
                                       parent[node], pushed, len(open_heap)))

    res.elapsed = time.perf_counter() - t0
    return res


def _reconstruct(parent: dict, goal: Node) -> list:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path
