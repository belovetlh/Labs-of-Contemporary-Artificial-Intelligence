"""通用工具：随机种子、设备选择、日志、JSON 读写、运行环境记录。"""
import json
import logging
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子。

    deterministic=True 时让 cuDNN 只使用确定性算法，牺牲少量速度换取可复现性。
    注意：不同显卡型号、CPU 与 GPU 之间的浮点结果仍可能有微小差异。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader 子进程的初始化函数，保证多进程下数据增强的随机性可复现。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """有 GPU 用 GPU，否则退回 CPU，保证同一份代码在两种环境下都能运行。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(log_file: Path) -> logging.Logger:
    """创建同时输出到控制台和文件的日志器（续训时以追加方式写入）。"""
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def env_info() -> dict:
    """记录运行环境，写入每次实验的结果文件，便于他人复现时核对。"""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
