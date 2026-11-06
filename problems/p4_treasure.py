"""题目四：竞争环境下的宝藏分配（A* + Alpha-Beta 综合题）。

两层结构：
  1. 底层用 A* 计算 A、B、T1..T4 两两之间的最小移动代价，得到距离矩阵 D；
  2. 上层把"轮流抢宝藏"建模为博弈树：每一步选一个尚未获得的宝藏 Ti，
     收益 Gain = V(Ti) - D[当前位置][Ti]，U = Score_A - Score_B，
     A(MAX) 最大化 U，B(MIN) 最小化 U。
距离一旦算好，博弈搜索中就只需查表，不必重复寻路。

约定（已与题目预期输出核对）：
  - 'X' 为障碍，数字为进入代价，A、B、T 所在格的进入代价为 1；起点格不计代价。
  - 深度 H 按层（ply）计：A 选一次是一层，B 选一次也是一层。H=4 即四个宝藏全部取完。
  - 每回合必须选一个宝藏，即使收益为负；到达深度上限或宝藏取完时，以当前 U 作为评价值。
  - 经过其他宝藏不会自动收集；已取走的宝藏格照常可通行；双方互不阻挡。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Optional

import time

from core.astar import SearchResult, astar
from core.game_search import (EXACT, INF, LOWER, UPPER, GameResult, GameStats, alphabeta,
                              minimax)
from core.grid import Coord, h_manhattan, h_zero, make_neighbors, min_step_cost

Dist = dict[str, dict[str, Optional[int]]]


def cell_cost(token: str):
    if token == "X":
        return None
    if token[0] in "ABT":
        return 1
    return int(token)


def point_key(name: str) -> tuple:
    """A、B 排在最前，宝藏按编号排序：A, B, T1, T2, ..."""
    return (0, 0) if name == "A" else (1, 0) if name == "B" else (2, int(name[1:]))


# ---------------------------------------------------------------- 第一层：A* 距离矩阵

def locate_points(grid: list[list[str]]) -> dict[str, Coord]:
    points = {tok: (r, c) for r, row in enumerate(grid) for c, tok in enumerate(row)
              if tok[0] in "ABT"}
    if "A" not in points or "B" not in points:
        raise ValueError("地图中必须包含 A 和 B")
    return dict(sorted(points.items(), key=lambda kv: point_key(kv[0])))


def pairwise_astar(grid: list[list[str]], points: dict[str, Coord],
                   heuristic: str = "manhattan") -> dict[tuple[str, str], SearchResult]:
    """对每个有序点对 (a, b), a != b 调用一次 A*，返回完整的搜索结果（含路径与统计）。

    进入代价只与目标格有关，d(p,q) 与 d(q,p) 在一般地图上不一定相等，因此所有有序对都要算。
    启发函数取 (最小进入代价 × 曼哈顿距离)：每步至少花这么多，所以可采纳且一致；
    heuristic="zero" 时退化为 Dijkstra，用于对照。
    """
    neighbors = make_neighbors(grid, cell_cost)
    w_min = min_step_cost(grid, cell_cost)
    out = {}
    for a, pa in points.items():
        for b, pb in points.items():
            if a != b:
                h = h_manhattan(pb, scale=w_min) if heuristic == "manhattan" else h_zero(pb)
                out[(a, b)] = astar(pa, pb, neighbors, h)
    return out


def distance_matrix(grid: list[list[str]], points: dict[str, Coord]) -> tuple[Dist, int]:
    """返回 (距离矩阵, A* 总扩展节点数)。不可达记为 None。"""
    runs = pairwise_astar(grid, points)
    dist: Dist = {a: {b: 0 if a == b else (runs[(a, b)].cost if runs[(a, b)].found else None)
                      for b in points} for a in points}
    return dist, sum(r.expanded for r in runs.values())


def format_matrix(dist: Dist) -> str:
    names = list(dist)
    lines = ["     " + "".join(f"{n:>5}" for n in names)]
    for a in names:
        cells = "".join(f"{'-' if dist[a][b] is None else dist[a][b]:>5}" for b in names)
        lines.append(f"{a:<5}" + cells)
    return "\n".join(lines)


# ---------------------------------------------------------------- 第二层：博弈建模

@dataclass(frozen=True)
class State:
    """博弈状态：双方位置、剩余宝藏、双方累计收益、当前玩家。

    位置用地点名记录（"A"、"T2"...），直接作为距离矩阵的下标。
    frozen + frozenset 使状态可哈希，后续做置换表/缓存时可以直接当字典键。
    """
    pos_a: str
    pos_b: str
    remaining: frozenset
    score_a: int
    score_b: int
    turn: str  # "A" 或 "B"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


class TreasureGame:
    """实现 core.game_search 的 Game 接口。"""

    def __init__(self, dist: Dist, values: dict[str, int]):
        self.dist = dist
        self.values = values

    def initial_state(self) -> State:
        return State("A", "B", frozenset(self.values), 0, 0, "A")

    def gain(self, s: State, t: str) -> int:
        pos = s.pos_a if s.turn == "A" else s.pos_b
        return self.values[t] - self.dist[pos][t]

    def actions(self, s: State) -> list[str]:
        pos = s.pos_a if s.turn == "A" else s.pos_b
        return [t for t in sorted(s.remaining, key=point_key) if self.dist[pos][t] is not None]

    def result(self, s: State, t: str) -> State:
        g = self.gain(s, t)
        rest = s.remaining - {t}
        if s.turn == "A":
            return replace(s, pos_a=t, score_a=s.score_a + g, remaining=rest, turn="B")
        return replace(s, pos_b=t, score_b=s.score_b + g, remaining=rest, turn="A")

    def is_max_turn(self, s: State) -> bool:
        return s.turn == "A"

    def evaluate(self, s: State) -> int:
        return s.utility


# ---------------------------------------------------------------- 求解与输出

def load(path: str) -> tuple[list[list[str]], dict[str, int], list[int]]:
    with open(path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    grid = [row.split() for row in cfg["grid"]]
    if len({len(r) for r in grid}) != 1:
        raise ValueError("地图各行长度不一致")
    return grid, cfg["values"], cfg.get("depths", [2, 4])


def solve(game: TreasureGame, depth: int, method: str = "alphabeta", order=None,
          cache: bool = False, state: State | None = None) -> GameResult:
    """用通用框架求解。cache=True 时置换表以完整状态（含累计收益）为键。"""
    algo = {"alphabeta": alphabeta, "minimax": minimax}[method]
    return algo(game, state or game.initial_state(), depth=depth, order=order, cache=cache)


def solve_incremental(game: TreasureGame, depth: int, prune: bool = True, cache: bool = True,
                      order=None, state: State | None = None) -> GameResult:
    """增量式搜索：每个节点返回"从这里往后还能让 U 变化多少"，而不是最终的 U。

    最终 U = 当前 U + 未来增量，而未来增量只取决于 (双方位置, 剩余宝藏, 当前玩家, 剩余深度)，
    与之前怎么攒下的累计收益无关。于是置换表的键可以去掉 score_a / score_b：
    "A 先拿 T1、B 再拿 T2" 与 "A 先拿 T2、B 再拿 T1" 之后若位置相同，就能复用同一个结果。

    窗口相应平移：父节点在增量空间的窗口为 (alpha, beta)，走一步使 U 变化 step，
    则子节点的窗口为 (alpha - step, beta - step)，子节点返回值加上 step 即为父节点视角的值。
    结果与 solve() 完全相同（测试中校验），只是开销更小。
    """
    stats = GameStats()
    tt: dict = {}
    t0 = time.perf_counter()

    def rec(s: State, d: int, alpha: float, beta: float, path: tuple):
        stats.nodes += 1
        key = (s.pos_a, s.pos_b, s.remaining, s.turn, d)
        if cache and key in tt:
            v, flag, pv = tt[key]
            if flag == LOWER:
                alpha = max(alpha, v)
            elif flag == UPPER:
                beta = min(beta, v)
            if flag == EXACT or alpha >= beta:
                stats.tt_hits += 1
                return v, pv
        acts = [] if d == 0 else game.actions(s)
        if not acts:
            stats.leaves += 1
            return 0, []
        if order:
            acts = order(game, s, acts)
        maximizing = s.turn == "A"
        a0, b0 = alpha, beta
        best, best_pv = (-INF if maximizing else INF), []
        for i, t in enumerate(acts):
            child = game.result(s, t)
            step = child.utility - s.utility
            v, pv = rec(child, d - 1, alpha - step, beta - step, path + (t,))
            v += step
            if (v > best) if maximizing else (v < best):
                best, best_pv = v, [t] + pv
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if prune and alpha >= beta:
                if i + 1 < len(acts):
                    stats.cutoffs += 1
                    stats.pruned.append((path, acts[i + 1:]))
                break
        if cache:
            flag = UPPER if best <= a0 else LOWER if best >= b0 else EXACT
            tt[key] = (best, flag, best_pv)
        return best, best_pv

    s0 = state or game.initial_state()
    delta, pv = rec(s0, depth, -INF, INF, ())
    stats.elapsed = time.perf_counter() - t0
    return GameResult(s0.utility + delta, pv, stats)


# ---------------------------------------------------------------- 走法排序

def order_reverse(game: TreasureGame, s: State, acts: list) -> list:
    return acts[::-1]


def order_greedy(game: TreasureGame, s: State, acts: list) -> list:
    """按当前玩家自己的即时收益从高到低访问：A 先看让 U 涨得多的，B 先看让 U 跌得多的。"""
    return sorted(acts, key=lambda t: game.gain(s, t), reverse=True)


def order_oracle(game: TreasureGame, total_depth: int, worst: bool = False):
    """理想（或最差）顺序：先算出每个子节点在剩余深度下的精确值再排序，排序开销不计入。

    剩余深度可由已取走的宝藏数推出：total_depth - (宝藏总数 - 剩余宝藏数)。
    """
    n = len(game.values)

    def order(g, s: State, acts: list) -> list:
        d = total_depth - (n - len(s.remaining)) - 1
        vals = {t: solve_incremental(g, d, state=g.result(s, t)).value for t in acts}
        best_first = sorted(acts, key=lambda t: vals[t], reverse=s.turn == "A")
        return best_first[::-1] if worst else best_first
    return order


# ---------------------------------------------------------------- 对局模拟

def play(game: TreasureGame, depth_a: int | None, depth_b: int | None) -> tuple[State, list[str]]:
    """双方各自用有限深度搜索、每回合重新决策，把整局走完。depth=None 表示看到终局。

    用来观察"视野"的影响：浅层搜索在当下看来最好的选择，放到整局里是否真的好。
    """
    s, moves = game.initial_state(), []
    while game.actions(s):
        d = depth_a if s.turn == "A" else depth_b
        d = len(s.remaining) if d is None else min(d, len(s.remaining))
        t = solve_incremental(game, d, state=s).best_action
        moves.append(f"{s.turn}->{t}")
        s = game.result(s, t)
    return s, moves


def answer(res: GameResult) -> dict:
    """题目要求的输出格式。"""
    return {"best_target": res.best_action, "best_value": res.value}


def explain_pv(game: TreasureGame, pv: list[str]) -> str:
    """把主变例展开成 "A:A->T2(6-4=+2)" 这样的逐步说明。"""
    s, steps = game.initial_state(), []
    for t in pv:
        pos = s.pos_a if s.turn == "A" else s.pos_b
        steps.append(f"{s.turn}:{pos}->{t}({game.values[t]}-{game.dist[pos][t]}={game.gain(s, t):+d})")
        s = game.result(s, t)
    return "  ".join(steps) + f"  => U={s.utility}"


def run(path: str) -> None:
    grid, values, depths = load(path)
    dist, expanded = distance_matrix(grid, locate_points(grid))
    print("距离矩阵（行=出发点，列=目标点）：")
    print(format_matrix(dist))
    print(f"A* 共扩展节点 {expanded}")
    game = TreasureGame(dist, values)
    for h in depths:
        for method in ("minimax", "alphabeta"):
            res = solve(game, h, method)
            print(f"H={h} {method:<9} => {json.dumps(answer(res))}"
                  f"  访问节点 {res.stats.nodes}，剪枝 {res.stats.cutoffs} 次")
        print("  主变例: " + explain_pv(game, res.pv))
