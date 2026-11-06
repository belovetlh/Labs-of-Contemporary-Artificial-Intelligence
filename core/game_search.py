"""通用博弈树搜索：Minimax 与 Alpha-Beta 剪枝。

问题通过实现 Game 接口接入（题目三的静态树、题目四的宝藏博弈都是如此）：
  - actions(state)     -> 当前状态下可选的走法列表；为空表示终局
  - result(state, a)   -> 执行走法 a 后的新状态
  - is_max_turn(state) -> 当前是否轮到 MAX
  - evaluate(state)    -> 叶子（终局或到达深度上限）的评价值
depth 表示剩余可搜索的层数（ply，每个玩家走一步算一层）；None 表示不限深度。

可选功能（默认关闭，不影响结果，只影响开销）：
  - order       ：走法排序，决定子节点的访问顺序
  - cache       ：置换表。同一状态经不同路径到达时（共享节点）直接复用结果；状态须可哈希
  - prune_equal ：剪枝条件取 alpha >= beta（默认）还是 alpha > beta
  - trace       ：记录每个访问节点的 (alpha, beta) 窗口、返回值与剪枝情况
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

INF = float("inf")


class Game(Protocol):
    def actions(self, state) -> Sequence: ...
    def result(self, state, action): ...
    def is_max_turn(self, state) -> bool: ...
    def evaluate(self, state) -> float: ...


# 走法排序策略：(game, state, actions) -> 重新排序后的 actions
OrderFn = Callable[[Game, Any, list], list]


@dataclass
class NodeTrace:
    """Alpha-Beta 中一个节点的访问记录，按访问（先序）顺序排列。"""
    index: int            # 第几个被访问的节点
    path: tuple           # 从根到该节点的走法序列
    is_max: bool
    alpha: float          # 进入该节点时的窗口
    beta: float
    value: float = None   # 回传给父节点的值
    leaf: bool = False
    cut: list = None      # 在该节点被剪掉的走法（没有剪枝则为 None）
    from_cache: bool = False


@dataclass
class GameStats:
    nodes: int = 0     # 访问的节点数（含根；置换表命中的节点也计入）
    leaves: int = 0    # 调用 evaluate 的次数
    cutoffs: int = 0   # 发生剪枝的次数
    tt_hits: int = 0   # 置换表命中次数
    # 剪枝记录：(从根到发生剪枝节点的走法序列, 被剪掉的走法列表)
    pruned: list = field(default_factory=list)
    trace: list = field(default_factory=list)  # trace=True 时为 NodeTrace 列表
    elapsed: float = 0.0


@dataclass
class GameResult:
    value: float          # 根节点的博弈值
    pv: list              # 主变例：双方都按最优策略走出的走法序列，pv[0] 即最优首步
    stats: GameStats

    @property
    def best_action(self):
        return self.pv[0] if self.pv else None


def _next_depth(depth: Optional[int]) -> Optional[int]:
    return None if depth is None else depth - 1


def minimax(game: Game, state, depth: Optional[int] = None,
            order: Optional[OrderFn] = None, cache: bool = False) -> GameResult:
    """朴素 Minimax：完整遍历博弈树（到 depth 为止），作为 Alpha-Beta 的对照基线。

    cache=True 时对 (状态, 剩余深度) 做记忆化：Minimax 的返回值总是精确值，可以直接复用。
    """
    stats = GameStats()
    memo: dict = {}
    t0 = time.perf_counter()

    def rec(s, d):
        stats.nodes += 1
        if cache and (s, d) in memo:
            stats.tt_hits += 1
            return memo[(s, d)]
        acts = [] if d == 0 else list(game.actions(s))
        if not acts:
            stats.leaves += 1
            out = game.evaluate(s), []
        else:
            if order:
                acts = order(game, s, acts)
            maximizing = game.is_max_turn(s)
            best, best_pv = (-INF if maximizing else INF), []
            for a in acts:
                v, pv = rec(game.result(s, a), _next_depth(d))
                # 严格比较：值相同时保留先访问到的走法，结果只依赖访问顺序，可复现
                if (v > best) if maximizing else (v < best):
                    best, best_pv = v, [a] + pv
            out = best, best_pv
        if cache:
            memo[(s, d)] = out
        return out

    value, pv = rec(state, depth)
    stats.elapsed = time.perf_counter() - t0
    return GameResult(value, pv, stats)


# 置换表条目的类型：EXACT 为精确值；LOWER/UPPER 表示真实值 >= / <= 存储值（因剪枝只得到了界）
EXACT, LOWER, UPPER = "exact", "lower", "upper"


def alphabeta(game: Game, state, depth: Optional[int] = None,
              order: Optional[OrderFn] = None, cache: bool = False,
              prune_equal: bool = True, trace: bool = False) -> GameResult:
    """Alpha-Beta 剪枝。

    alpha：从根到当前节点的路径上，MAX 已经能保证拿到的最低分（下界）
    beta ：同一路径上，MIN 已经能保证把分数压到的最高分（上界）
    一旦 alpha >= beta，当前节点剩下的子节点无论取什么值都不会影响根节点的结果，于是剪掉。
    根节点用窗口 (-inf, +inf) 调用，返回值与 Minimax 完全相同。

    带置换表时要注意：被剪枝的节点返回的只是一个界而不是精确值，
    所以条目要同时记下类型（精确/下界/上界），复用时据此收紧或直接返回。
    """
    stats = GameStats()
    tt: dict = {}  # (状态, 剩余深度) -> (值, 类型, 主变例)
    t0 = time.perf_counter()

    def rec(s, d, alpha, beta, path):
        stats.nodes += 1
        rec_ = None
        if trace:
            rec_ = NodeTrace(stats.nodes, path, game.is_max_turn(s), alpha, beta)
            stats.trace.append(rec_)

        key = (s, d)
        if cache and key in tt:
            v, flag, pv = tt[key]
            if flag == LOWER:
                alpha = max(alpha, v)
            elif flag == UPPER:
                beta = min(beta, v)
            if flag == EXACT or alpha >= beta:
                stats.tt_hits += 1
                if rec_:
                    rec_.value, rec_.from_cache = v, True
                return v, pv

        acts = [] if d == 0 else list(game.actions(s))
        if not acts:
            stats.leaves += 1
            v = game.evaluate(s)
            if cache:
                tt[key] = (v, EXACT, [])
            if rec_:
                rec_.value, rec_.leaf = v, True
            return v, []

        if order:
            acts = order(game, s, acts)
        maximizing = game.is_max_turn(s)
        a0, b0 = alpha, beta  # 实际搜索所用的窗口，用来判断结果是精确值还是界
        best, best_pv = (-INF if maximizing else INF), []
        for i, a in enumerate(acts):
            v, pv = rec(game.result(s, a), _next_depth(d), alpha, beta, path + (a,))
            if maximizing:
                if v > best:
                    best, best_pv = v, [a] + pv
                alpha = max(alpha, best)
            else:
                if v < best:
                    best, best_pv = v, [a] + pv
                beta = min(beta, best)
            if alpha >= beta if prune_equal else alpha > beta:
                if i + 1 < len(acts):  # 后面还有子节点没访问，才算真正剪掉了东西
                    stats.cutoffs += 1
                    stats.pruned.append((path, acts[i + 1:]))
                    if rec_:
                        rec_.cut = acts[i + 1:]
                break

        if cache:
            flag = UPPER if best <= a0 else LOWER if best >= b0 else EXACT
            tt[key] = (best, flag, best_pv)
        if rec_:
            rec_.value = best
        return best, best_pv

    value, pv = rec(state, depth, -INF, INF, ())
    stats.elapsed = time.perf_counter() - t0
    return GameResult(value, pv, stats)
