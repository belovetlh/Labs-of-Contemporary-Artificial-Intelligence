"""题目三实验用的博弈树生成器与子节点排序策略。

所有树都由 seed 唯一确定。根节点为 MAX，逐层交替（与题目约定一致）。
"""
from __future__ import annotations

import math
import random

from problems.p3_tree import TreeGame, TreeNode


# ---------------------------------------------------------------- 生成

def random_tree(b: int, d: int, seed: int, int_values: tuple | None = None) -> TreeNode:
    """分支数 b、深度 d 的满树，叶子值独立随机。

    默认叶子取 [0,1) 的实数（几乎不会相等，便于和理论公式对照）；
    int_values=(lo, hi) 时取整数，会出现大量相等的值。
    """
    rng = random.Random(seed)

    def leaf():
        return rng.randint(*int_values) if int_values else rng.random()

    def build(depth):
        if depth == 0:
            return TreeNode(value=leaf())
        return TreeNode(children=[build(depth - 1) for _ in range(b)])
    return build(d)


def correlated_tree(b: int, d: int, seed: int, sigma: float = 1.0) -> TreeNode:
    """子节点价值与父节点相关的树，更接近真实棋类。

    每个节点有一个"局面估值" est：子节点 est = 父节点 est + N(0, sigma)，叶子值即其 est。
    于是内部节点的 est 是其真实博弈值的一个带噪声的近似，可以拿来给子节点排序。
    """
    rng = random.Random(seed)

    def build(depth, est):
        node = TreeNode(value=est if depth == 0 else None)
        node.est = est
        if depth > 0:
            node.children = [build(depth - 1, est + rng.gauss(0, sigma)) for _ in range(b)]
        return node
    return build(d, 0.0)


def layered_dag(depth: int, width: int, b: int, seed: int) -> TreeNode:
    """分层有向无环图：每层 width 个节点，每个节点从下一层随机选 b 个作为子节点。

    展开成树有 b^depth 片叶子，但不同的节点只有约 depth*width 个——大量共享节点，
    相当于棋类中"不同走法顺序到达同一局面"（置换）。
    """
    rng = random.Random(seed)
    layer = [TreeNode(value=rng.random()) for _ in range(width)]
    for _ in range(depth - 1):
        layer = [TreeNode(children=rng.sample(layer, b)) for _ in range(width)]
    return TreeNode(children=rng.sample(layer, b))


def knuth_moore_leaves(b: int, d: int) -> int:
    """完美排序时 Alpha-Beta 评估的叶子数（Knuth & Moore, 1975）：b^ceil(d/2) + b^floor(d/2) - 1。"""
    return b ** math.ceil(d / 2) + b ** (d // 2) - 1


# ---------------------------------------------------------------- 真实值与排序

def exact_values(root: TreeNode) -> dict:
    """每个 (节点, 是否 MAX) 的精确博弈值，供"理想排序"使用（排序本身的开销不计入）。"""
    memo = {}

    def rec(node, is_max):
        key = (id(node), is_max)
        if key not in memo:
            if not node.children:
                memo[key] = node.value
            else:
                vals = [rec(c, not is_max) for c in node.children]
                memo[key] = max(vals) if is_max else min(vals)
        return memo[key]
    rec(root, True)
    return memo


def order_by(score):
    """生成排序函数：MAX 节点按 score 从高到低访问子节点，MIN 节点从低到高。

    score(child_node, child_is_max) -> 数值，越大表示对 MAX 越有利。
    """
    def order(game: TreeGame, state, actions):
        node, is_max = state
        return sorted(actions, key=lambda i: score(node.children[i], not is_max), reverse=is_max)
    return order


def order_best(values: dict):
    """理想排序：每层先访问真正最好的子节点。"""
    return order_by(lambda c, m: values[(id(c), m)])


def order_worst(values: dict):
    """最差排序：每层先访问真正最差的子节点。"""
    best = order_best(values)
    return lambda game, state, actions: best(game, state, actions)[::-1]


def order_reverse(game, state, actions):
    """从右到左。"""
    return list(actions)[::-1]


def noisy_estimates(root: TreeNode, tau: float, seed: int) -> dict:
    """给 correlated_tree 的每个节点的 est 再加一层 N(0, tau) 噪声，模拟不准确的估值函数。

    噪声用独立的随机数流：若与建树共用同一个 seed，两边抽到的随机数序列会对齐，
    噪声就和树结构相关了（曾因此出现"比随机猜还差"的命中率）。
    各 tau 共用同一组标准正态噪声再按 tau 缩放，使不同 tau 之间的比较只差在噪声大小上。
    """
    rng = random.Random(f"noise-{seed}")
    est = {}

    def rec(node):
        if id(node) not in est:
            est[id(node)] = node.est + tau * rng.gauss(0, 1)
            for c in node.children:
                rec(c)
    rec(root)
    return est


def order_by_estimate(est: dict):
    return order_by(lambda c, m: est[id(c)])


def first_move_accuracy(root: TreeNode, values: dict, est: dict) -> float:
    """按估值排在第一位的子节点恰好是真正最优子节点的比例（衡量排序质量）。"""
    hit = total = 0
    seen = set()

    def rec(node, is_max):
        nonlocal hit, total
        if not node.children or (id(node), is_max) in seen:
            return
        seen.add((id(node), is_max))
        kids = node.children
        pick = (max if is_max else min)(kids, key=lambda c: est[id(c)])
        true = (max if is_max else min)(values[(id(c), not is_max)] for c in kids)
        hit += values[(id(pick), not is_max)] == true
        total += 1
        for c in kids:
            rec(c, not is_max)
    rec(root, True)
    return hit / total
