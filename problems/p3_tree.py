"""题目三：极大极小博弈树的最佳得分（Alpha-Beta 剪枝）。

TreeNode 与题目给出的定义一致：非叶子节点有 children，叶子节点有 value。
根节点是 MAX，之后逐层 MAX/MIN 交替；子节点从左到右访问。

支持两种输入：
  1. 直接用 TreeNode 构建的树（题目"构建代码示例"）；
  2. 带 is_maximizing_player 字段的 JSON（题目"输入样例"）。
JSON 中的 is_maximizing_player 会挂到节点的 is_max 属性上；搜索时优先使用该属性，
没有该属性时按层交替推断。两种来源因此可以共用同一套搜索代码。
"""
from __future__ import annotations

import json

from core.game_search import GameResult, alphabeta, minimax


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value  # 叶子节点的得分
        self.children = children if children is not None else []  # 子节点列表


class TreeGame:
    """把 TreeNode 树适配为 core.game_search 的 Game 接口。

    状态为 (节点, 该节点是否为 MAX)；走法为子节点下标 0, 1, 2, ...
    """

    def initial_state(self, root: TreeNode):
        flag = getattr(root, "is_max", None)
        return (root, True if flag is None else flag)

    def actions(self, state):
        node, _ = state
        return range(len(node.children))

    def result(self, state, i):
        node, is_max = state
        child = node.children[i]
        flag = getattr(child, "is_max", None)
        return (child, (not is_max) if flag is None else flag)

    def is_max_turn(self, state) -> bool:
        return state[1]

    def evaluate(self, state):
        node, _ = state
        if node.value is None:
            raise ValueError("叶子节点缺少 value")
        return node.value


# ---------------------------------------------------------------- 构建与解析

def tree_from_dict(d: dict) -> TreeNode:
    """JSON 字典 -> TreeNode。可以带最外层的 "tree_structure"，也可以不带。"""
    if "tree_structure" in d:
        d = d["tree_structure"]
    children = [tree_from_dict(c) for c in d.get("children", [])]
    node = TreeNode(value=d.get("value"), children=children)
    if "is_maximizing_player" in d:
        node.is_max = bool(d["is_maximizing_player"])
    return node


def load_tree(path: str) -> TreeNode:
    with open(path, encoding="utf-8-sig") as f:
        return tree_from_dict(json.load(f))


def build_sample_tree() -> TreeNode:
    """照抄题目的构建代码。注意 max2 同时是 min1 和 min2 的子节点（共享节点）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


# ---------------------------------------------------------------- 求解

def search(root: TreeNode, method: str = "alphabeta", order=None) -> GameResult:
    game = TreeGame()
    algo = {"alphabeta": alphabeta, "minimax": minimax}[method]
    return algo(game, game.initial_state(root), order=order)


def alpha_beta(root: TreeNode):
    """题目要求的接口：接收根节点，返回对 MAX 而言的最佳得分。"""
    return search(root, "alphabeta").value


def describe_pruned(root: TreeNode, res: GameResult) -> list[str]:
    """把剪枝记录翻译成可读文字，例如 "root-0-1 剪掉子节点[1]（叶子值 [9]）"。"""
    lines = []
    for path, cut in res.stats.pruned:
        node = root
        for i in path:
            node = node.children[i]
        leaves = [_leaf_values(node.children[i]) for i in cut]
        label = "-".join(["root", *map(str, path)])
        lines.append(f"{label} 剪掉子节点{list(cut)}（叶子值 {sum(leaves, [])}）")
    return lines


def _leaf_values(node: TreeNode) -> list:
    if not node.children:
        return [node.value]
    return sum((_leaf_values(c) for c in node.children), [])


def run(path: str) -> None:
    root = load_tree(path)
    ab = search(root, "alphabeta")
    mm = search(root, "minimax")
    print(ab.value)
    print(f"Minimax   : 值 {mm.value}，访问节点 {mm.stats.nodes}，叶子 {mm.stats.leaves}")
    print(f"Alpha-Beta: 值 {ab.value}，访问节点 {ab.stats.nodes}，叶子 {ab.stats.leaves}，"
          f"剪枝 {ab.stats.cutoffs} 次")
    for line in describe_pruned(root, ab):
        print("  " + line)
