"""实验二统一入口。

用法：
  python main.py p1 [输入文件]     # 默认 data/p1_sample.txt
  python main.py p2 [输入文件]
  python main.py p3 [输入文件]
  python main.py p4 [输入文件]
  python main.py all               # 依次运行四道题的样例
"""
import argparse
import os
import sys

from problems import p1_maze, p2_weighted, p3_tree, p4_treasure

ROOT = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = {
    "p1": (p1_maze.run, "data/p1_sample.txt", "题目一 迷宫寻路"),
    "p2": (p2_weighted.run, "data/p2_sample.txt", "题目二 加权地形"),
    "p3": (p3_tree.run, "data/p3_sample.json", "题目三 Alpha-Beta 剪枝"),
    "p4": (p4_treasure.run, "data/p4_sample.json", "题目四 宝藏分配"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="当代人工智能 实验二：搜索算法")
    parser.add_argument("problem", choices=[*PROBLEMS, "all"])
    parser.add_argument("input", nargs="?", help="输入文件路径，缺省时使用样例")
    args = parser.parse_args()
    # Windows 下输出被重定向到文件或管道时默认用 GBK，统一为 UTF-8 避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")

    names = list(PROBLEMS) if args.problem == "all" else [args.problem]
    for name in names:
        run, default, title = PROBLEMS[name]
        path = args.input if args.input and args.problem != "all" else os.path.join(ROOT, default)
        print(f"===== {title}  ({os.path.relpath(path, ROOT)}) =====")
        run(path)
        print()


if __name__ == "__main__":
    main()
