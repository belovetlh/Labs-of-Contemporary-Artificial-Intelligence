"""题目给出的测试用例 + 若干边界情况。运行：python -m unittest -v"""
import os
import subprocess
import sys
import tempfile
import unittest

from core.grid import manhattan, parse_grid
from core.game_search import alphabeta, minimax
from experiments.admissibility import audit, true_cost_to_goal
from experiments.analysis import cheap_connected, classify_by_f, min_leaves
from experiments.common import paired_test
from experiments.game_trees import (exact_values, knuth_moore_leaves, layered_dag, order_best,
                                    order_worst, random_tree)
from experiments.maze_gen import random_grid, treasure_map, weighted_grid
from problems import p1_maze, p2_weighted, p3_tree, p4_treasure

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def data(name):
    return os.path.join(DATA, name)


class TestP1Maze(unittest.TestCase):
    def assert_valid_path(self, path, start, goal, grid):
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        for a, b in zip(path, path[1:]):
            self.assertEqual(manhattan(a, b), 1, "相邻两步必须是上下左右一格")
        for r, c in path:
            self.assertNotEqual(grid[r][c], "X", "路径不能穿过障碍")

    def test_sample(self):
        start, goal, grid = p1_maze.load_grid_problem(data("p1_sample.txt"))
        path = p1_maze.find_path(start, goal, grid)
        # 最短路可能不唯一，所以校验合法性与长度，而不是逐点比对
        self.assert_valid_path(path, start, goal, grid)
        self.assertEqual(len(path), 8)

    def test_no_path(self):
        start, goal, grid = p1_maze.load_grid_problem(data("p1_nopath.txt"))
        self.assertEqual(p1_maze.find_path(start, goal, grid), [])

    def test_start_equals_goal(self):
        self.assertEqual(p1_maze.find_path((0, 0), (0, 0), [["S"]]), [(0, 0)])

    def test_heuristics_agree_on_length(self):
        start, goal, grid = p1_maze.load_grid_problem(data("p1_sample.txt"))
        lengths = {h: len(p1_maze.solve(start, goal, grid, heuristic=h).path)
                   for h in p1_maze.HEURISTICS}
        self.assertEqual(set(lengths.values()), {8})

    def test_start_or_goal_on_wall(self):
        grid = [[".", "X"], [".", "."]]
        self.assertEqual(p1_maze.find_path((0, 1), (1, 1), grid), [])
        self.assertEqual(p1_maze.find_path((0, 0), (0, 1), grid), [])

    def test_out_of_bounds(self):
        grid = [[".", "."], [".", "."]]
        self.assertEqual(p1_maze.find_path((0, 0), (2, 0), grid), [])
        self.assertEqual(p1_maze.find_path((-1, 0), (1, 1), grid), [])

    def test_grid_without_s_and_e(self):
        # 只依赖输入坐标，地图上没有 S/E 字符也能求解
        grid = [[".", ".", "."], ["X", "X", "."], [".", ".", "."]]
        path = p1_maze.find_path((0, 0), (2, 0), grid)
        self.assert_valid_path(path, (0, 0), (2, 0), grid)
        self.assertEqual(len(path), 7)

    def test_random_maps_all_heuristics_optimal(self):
        for seed in range(10):
            grid, s, g = random_grid(25, 0.3, seed)
            lengths = {len(p1_maze.solve(s, g, grid, heuristic=h).path) for h in p1_maze.HEURISTICS}
            self.assertEqual(len(lengths), 1, f"seed={seed} 各启发函数的最短路长度不一致")

    def test_trace_consistent_heuristic(self):
        # 启发函数一致时，出队节点的 f 值单调不减；trace 的步数等于展开数
        grid, s, g = random_grid(25, 0.2, 0)
        res = p1_maze.solve(s, g, grid, trace=True)
        self.assertEqual(len(res.steps), res.expanded)
        fs = [st.f for st in res.steps]
        self.assertEqual(fs, sorted(fs))
        self.assertEqual(res.reopened, 0)

    def test_maze_gen_reproducible(self):
        self.assertEqual(random_grid(15, 0.25, 7), random_grid(15, 0.25, 7))


class TestP2Weighted(unittest.TestCase):
    def test_sample(self):
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_sample.txt"))
        self.assertEqual(p2_weighted.min_cost(start, goal, grid), 7)

    def test_all_heuristics(self):
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_sample.txt"))
        for h in p2_weighted.HEURISTICS:
            self.assertEqual(p2_weighted.solve(start, goal, grid, heuristic=h).cost, 7, h)

    def test_no_path(self):
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_nopath.txt"))
        self.assertEqual(p2_weighted.min_cost(start, goal, grid), -1)

    def test_breakdown_sums_to_cost(self):
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_sample.txt"))
        res = p2_weighted.solve(start, goal, grid)
        self.assertEqual(p2_weighted.cost_breakdown(grid, res.path)[-1][2], res.cost)

    def test_unquoted_tokens(self):
        self.assertEqual(parse_grid("[[S, 1], [8, E]]"), [["S", "1"], ["8", "E"]])

    def test_true_cost_matches_answer(self):
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_sample.txt"))
        self.assertEqual(true_cost_to_goal(grid, goal, p2_weighted.cell_cost)[start], 7)

    def test_manhattan_overestimates_on_sample(self):
        # 进入 E 无代价，曼哈顿在样例上高估（含起点：h(S)=8 > h*(S)=7），但仍求得 7
        start, goal, grid = p2_weighted.load_grid_problem(data("p2_sample.txt"))
        h = p2_weighted.make_heuristic("manhattan", goal, grid)
        self.assertEqual(h(start), 8)
        self.assertGreater(audit(grid, goal, h, p2_weighted.cell_cost)["over_nodes"], 0)

    def test_goal_aware_admissible_and_consistent(self):
        for seed in range(10):
            grid, s, g = weighted_grid(20, seed, p_high=0.5, density=0.1)
            h = p2_weighted.make_heuristic("goal_aware", g, grid)
            a = audit(grid, g, h, p2_weighted.cell_cost)
            self.assertEqual((a["over_nodes"], a["inconsistent_edges"]), (0, 0), f"seed={seed}")

    def test_excluding_start_strengthens_heuristic(self):
        # 最便宜的地形代价为 3 时，w_min 排除起点后启发更强、展开更少，且仍然最优、不高估
        for seed in range(5):
            grid, s, g = weighted_grid(20, seed, p_high=0.5, low=3, high=(5, 9))
            old = p2_weighted.solve(s, g, grid, "goal_aware_incl_start")
            new = p2_weighted.solve(s, g, grid, "goal_aware")
            self.assertEqual(new.cost, p2_weighted.solve(s, g, grid, "zero").cost)
            self.assertEqual(new.cost, old.cost)
            self.assertLess(new.expanded, old.expanded)
            h = p2_weighted.make_heuristic("goal_aware", g, grid, s)
            self.assertEqual(audit(grid, g, h, p2_weighted.cell_cost, blocked=(s,))["over_nodes"], 0)

    def test_random_maps_optimal(self):
        for seed in range(30):
            grid, s, g = weighted_grid(15, seed, p_high=0.5, density=0.1)
            costs = {p2_weighted.solve(s, g, grid, h).cost for h in p2_weighted.HEURISTICS}
            self.assertEqual(len(costs), 1, f"seed={seed}")

    def test_weighted_grid_reproducible(self):
        self.assertEqual(weighted_grid(12, 3, p_high=0.4), weighted_grid(12, 3, p_high=0.4))


class TestP3Tree(unittest.TestCase):
    def test_treenode_sample(self):
        self.assertEqual(p3_tree.alpha_beta(p3_tree.build_sample_tree()), 5)

    def test_json_sample(self):
        self.assertEqual(p3_tree.alpha_beta(p3_tree.load_tree(data("p3_sample.json"))), 5)

    def test_minimax_matches(self):
        root = p3_tree.build_sample_tree()
        self.assertEqual(p3_tree.search(root, "minimax").value, 5)

    def test_pruning(self):
        # min1 已有 5，max2 读到 6 >= 5 后，叶子 9 被剪掉
        res = p3_tree.search(p3_tree.build_sample_tree(), "alphabeta")
        self.assertEqual(res.stats.pruned, [((0, 1), [1])])

    def test_single_leaf(self):
        self.assertEqual(p3_tree.alpha_beta(p3_tree.TreeNode(value=42)), 42)

    def test_trace_matches_stats(self):
        game = p3_tree.TreeGame()
        res = alphabeta(game, game.initial_state(p3_tree.build_sample_tree()), trace=True)
        self.assertEqual(len(res.stats.trace), res.stats.nodes)
        cut = [t for t in res.stats.trace if t.cut]
        self.assertEqual([(t.path, t.cut) for t in cut], [((0, 1), [1])])

    def test_all_options_same_value(self):
        # 访问顺序、置换表、剪枝条件只影响开销，不影响根节点的值
        game = p3_tree.TreeGame()
        for seed in range(15):
            root = random_tree(3, 5, seed, int_values=(0, 5))
            s = game.initial_state(root)
            expected = minimax(game, s).value
            v = exact_values(root)
            for kw in ({}, {"order": order_best(v)}, {"order": order_worst(v)}, {"cache": True},
                       {"prune_equal": False}, {"order": order_best(v), "cache": True}):
                self.assertEqual(alphabeta(game, s, **kw).value, expected, (seed, kw))

    def test_ideal_order_matches_knuth_moore(self):
        game = p3_tree.TreeGame()
        for b, d in ((2, 6), (3, 5), (4, 4)):
            root = random_tree(b, d, 0)
            res = alphabeta(game, game.initial_state(root), order=order_best(exact_values(root)))
            self.assertEqual(res.stats.leaves, knuth_moore_leaves(b, d), (b, d))

    def test_cache_on_dag(self):
        game = p3_tree.TreeGame()
        for seed in range(5):
            root = layered_dag(8, 5, 3, seed)
            s = game.initial_state(root)
            plain = minimax(game, s)
            for algo in (minimax, alphabeta):
                cached = algo(game, s, cache=True)
                self.assertEqual(cached.value, plain.value)
                self.assertLess(cached.stats.nodes, plain.stats.nodes)


class TestP4Treasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grid, values, _ = p4_treasure.load(data("p4_sample.json"))
        cls.dist, _ = p4_treasure.distance_matrix(grid, p4_treasure.locate_points(grid))
        cls.game = p4_treasure.TreasureGame(cls.dist, values)

    def test_distance_matrix(self):
        names = ["A", "B", "T1", "T2", "T3", "T4"]
        expected = [
            [0, 12, 10, 4, 8, 8],
            [12, 0, 8, 8, 4, 10],
            [10, 8, 0, 6, 6, 12],
            [4, 8, 6, 0, 6, 6],
            [8, 4, 6, 6, 0, 6],
            [8, 10, 12, 6, 6, 0],
        ]
        self.assertEqual([[self.dist[a][b] for b in names] for a in names], expected)

    def test_expected_answers(self):
        for h, target, value in ((2, "T2", -1), (4, "T3", 3)):
            for method in ("minimax", "alphabeta"):
                res = p4_treasure.solve(self.game, h, method)
                self.assertEqual(p4_treasure.answer(res),
                                 {"best_target": target, "best_value": value}, (h, method))

    def test_alphabeta_visits_fewer(self):
        mm = p4_treasure.solve(self.game, 4, "minimax")
        ab = p4_treasure.solve(self.game, 4, "alphabeta")
        self.assertLess(ab.stats.nodes, mm.stats.nodes)

    def test_dijkstra_gives_same_matrix(self):
        grid, _, _ = p4_treasure.load(data("p4_sample.json"))
        points = p4_treasure.locate_points(grid)
        a = p4_treasure.pairwise_astar(grid, points, "manhattan")
        z = p4_treasure.pairwise_astar(grid, points, "zero")
        self.assertEqual({k: r.cost for k, r in a.items()}, {k: r.cost for k, r in z.items()})

    def test_incremental_and_orders_match(self):
        # 增量式搜索、各种访问顺序、置换表只影响开销，答案与基准一致
        for h in (1, 2, 3, 4):
            ref = p4_treasure.answer(p4_treasure.solve(self.game, h, "minimax"))
            for res in (p4_treasure.solve_incremental(self.game, h),
                        p4_treasure.solve_incremental(self.game, h, prune=False),
                        p4_treasure.solve(self.game, h, cache=True),
                        p4_treasure.solve(self.game, h, order=p4_treasure.order_oracle(self.game, h))):
                self.assertEqual(res.value, ref["best_value"], h)
            # 无并列最优时，首步也必须一致
            self.assertEqual(p4_treasure.solve_incremental(self.game, h).best_action, ref["best_target"])

    def test_random_maps_incremental(self):
        for seed in range(5):
            grid, values = treasure_map(8, 6, seed)
            dist, _ = p4_treasure.distance_matrix(grid, p4_treasure.locate_points(grid))
            game = p4_treasure.TreasureGame(dist, values)
            ref = p4_treasure.solve(game, 6, "minimax").value
            for order in (None, p4_treasure.order_greedy, p4_treasure.order_reverse):
                self.assertEqual(p4_treasure.solve_incremental(game, 6, order=order).value, ref, seed)
                self.assertEqual(p4_treasure.solve(game, 6, order=order, cache=True).value, ref, seed)

    def test_optimal_play_reaches_game_value(self):
        # 双方都看到终局并按最优策略走完，最终 U 应等于全深度搜索的博弈值
        final, moves = p4_treasure.play(self.game, None, None)
        self.assertEqual(final.utility, 3)
        self.assertEqual(moves[0], "A->T3")


class TestAnalysisExperiments(unittest.TestCase):
    """分析实验所依赖的理论关系。"""

    def test_min_leaves_formula(self):
        # 理想顺序的叶子数等于广义 Knuth-Moore 公式（样例 H=2,3,4）
        grid, values, _ = p4_treasure.load(data("p4_sample.json"))
        dist, _ = p4_treasure.distance_matrix(grid, p4_treasure.locate_points(grid))
        game = p4_treasure.TreasureGame(dist, values)
        for h, expected in ((2, 6), (3, 10), (4, 10)):
            self.assertEqual(min_leaves([4 - i for i in range(h)]), expected)
            res = p4_treasure.solve(game, h, order=p4_treasure.order_oracle(game, h))
            self.assertEqual(res.stats.leaves, expected, h)

    def test_loss_from_first_move_only_when_later_moves_see_end(self):
        # 6 个宝藏、H>=4：第二、三次决策已能看到终局，总损失恰等于第一步造成的损失
        for seed in range(10):
            grid, values = treasure_map(10, 6, seed)
            dist, _ = p4_treasure.distance_matrix(grid, p4_treasure.locate_points(grid))
            game = p4_treasure.TreasureGame(dist, values)
            v = p4_treasure.solve_incremental(game, 6).value
            s0 = game.initial_state()
            for h in (4, 5):
                t = p4_treasure.solve_incremental(game, h).best_action
                first = v - p4_treasure.solve_incremental(game, 5, state=game.result(s0, t)).value
                total = v - p4_treasure.play(game, h, None)[0].utility
                self.assertEqual(total, first, (seed, h))

    def testcheap_connected(self):
        grid = [["S", "1", "8"], ["8", "1", "1"], ["8", "8", "E"]]
        self.assertTrue(cheap_connected(grid, (0, 0), (2, 2), monotone=True))
        grid = [["S", "8", "1"], ["1", "8", "1"], ["1", "1", "1"], ["8", "8", "E"]]
        self.assertTrue(cheap_connected(grid, (0, 0), (3, 2), monotone=False))
        self.assertTrue(cheap_connected(grid, (0, 0), (3, 2), monotone=True))
        grid = [["S", "1", "1"], ["8", "8", "1"], ["1", "1", "1"], ["1", "8", "8"], ["1", "1", "E"]]
        self.assertTrue(cheap_connected(grid, (0, 0), (4, 2), monotone=False))
        self.assertFalse(cheap_connected(grid, (0, 0), (4, 2), monotone=True))

    def test_f_decomposition_matches_theory(self):
        # f < C* 的节点恰好全部展开，f > C* 的一个也不展开（与平局规则无关）
        for seed in range(5):
            grid, s, g = random_grid(30, 0.3, seed)
            cls = classify_by_f(grid, s, g, p1_maze.cell_cost)
            for tie_break in ("low_h", "fifo", "lifo"):
                res = p1_maze.solve(s, g, grid, tie_break=tie_break, trace=True)
                self.assertEqual({st.node for st in res.steps if st.f < cls["cstar"]}, cls["must"])
                self.assertTrue(all(st.f <= cls["cstar"] for st in res.steps))

    def test_tests_do_not_need_plotting_libraries(self):
        # 解题代码和单元测试只依赖标准库：屏蔽画图库后仍应能导入
        code = "import sys; sys.modules['matplotlib'] = None; sys.modules['numpy'] = None; import tests.test_samples, main"
        root = os.path.dirname(DATA)
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bom_input(self):
        # Windows 记事本保存的文件可能带 BOM
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "t.json")
            with open(j, "w", encoding="utf-8-sig") as f:
                f.write('{"tree_structure": {"children": [{"value": 1}, {"value": 2}]}}')
            self.assertEqual(p3_tree.alpha_beta(p3_tree.load_tree(j)), 2)


class TestPairedTest(unittest.TestCase):
    def test_known_values(self):
        # 差值 1,2,3,4：均值 2.5，标准差 1.291，标准误 0.645，t = 3.873
        r = paired_test([0, 0, 0, 0], [1, 2, 3, 4])
        self.assertAlmostEqual(r["diff"], 2.5)
        self.assertAlmostEqual(r["se"], 1.2909944 / 2, places=6)
        self.assertAlmostEqual(r["t"], 3.8729833, places=6)
        self.assertTrue(r["significant"])

    def test_no_difference(self):
        r = paired_test([3, 5, 7], [3, 5, 7])
        self.assertEqual((r["diff"], r["t"], r["significant"]), (0, 0.0, False))

    def test_noise_not_significant(self):
        # 差值正负相抵、平均很小：不显著
        self.assertFalse(paired_test([0] * 6, [5, -4, 3, -5, 4, -2])["significant"])


if __name__ == "__main__":
    unittest.main()
