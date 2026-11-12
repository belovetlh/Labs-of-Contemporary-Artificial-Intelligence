# 实验三：图像分类

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/dataset-CIFAR--10-blue)

当代人工智能实验课 · 实验三。在 **CIFAR-10** 上完成图像分类：自行设计一个基线 CNN，在此基础上引入残差连接与注意力机制（SE / CBAM），并微调 ImageNet 预训练模型（ResNet-18 / MobileNetV2 / ViT-B/16）；用对照实验比较学习率调度、数据增强和损失函数的影响，用 Grad-CAM 分析模型关注的区域。

## 复现步骤

环境：Python 3.13.2，PyTorch 2.10（要求 PyTorch 2.3 及以上）。有 NVIDIA GPU 时自动使用 GPU 和混合精度，否则自动退回 CPU，代码无需修改。CIFAR-10 在首次运行时由 `torchvision` 自动下载到 `data/`（约 170 MB）。

**1. 获取代码并安装依赖**

```bash
git clone -b lab3 https://github.com/belovetlh/Labs-of-Contemporary-Artificial-Intelligence.git
```

```bash
cd Labs-of-Contemporary-Artificial-Intelligence
```

```bash
pip install -r requirements.txt
```

在 Windows 上 `pip` 默认安装 CPU 版 PyTorch；需要 GPU 版时按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的说明安装与 CUDA 版本对应的包。

**2. 快速检查**（只用 512 张图跑通"训练 → 验证 → 测试 → 出图"全流程，CPU 上不到一分钟，准确率没有参考意义）

```bash
python train.py --name quick_check --model simple_cnn --debug --epochs 2 --workers 0
```

**3. 训练单个模型**（以自行设计的基线 CNN 为例）

```bash
python train.py --name arch_simple_cnn --model simple_cnn --dropout 0.3
```

**4. 重新生成全部实验结果**（需要 GPU；分组运行，已完成的实验自动跳过，中断的实验从断点继续）

```bash
python run_experiments.py --group all
```

```bash
python compare.py --group all
```

在 Kaggle 上运行的完整步骤见 [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb)。

**5. 评估与可解释性分析**（需要对应实验目录下的 `best.pt`）

```bash
python evaluate.py --run outputs/base_resnet18
```

```bash
python explain.py --run outputs/base_resnet18
```

随机种子、数据划分和全部超参数都是固定的，每次实验的完整配置与运行环境随结果一起保存在 `outputs/<实验名>/` 下。不同 GPU 型号和 PyTorch 版本之间浮点运算有细微差异，重新训练得到的准确率可能与报告值相差零点几个百分点。

## 结论

待全部实验完成后填入。

## 用法

### 训练

```bash
python train.py --name my_run --model resnet18_se --aug autoaugment --label-smoothing 0.1
```

| 参数 | 可选值 | 默认 |
|---|---|---|
| `--model` | `simple_cnn` `resnet18` `resnet18_se` `resnet18_cbam` `resnet18_pt` `mobilenet_v2_pt` `vit_b_16_pt` | `resnet18` |
| `--aug` | `none` `basic` `cutout` `autoaugment` `randaugment` | `basic` |
| `--mixup-alpha` `--cutmix-alpha` | 大于 0 时启用 | 0 |
| `--loss` | `ce` `focal`，另有 `--label-smoothing` | `ce` |
| `--scheduler` | `none` `step` `cosine` `onecycle` `plateau`，另有 `--warmup-epochs` | `cosine` |
| `--early-stop-patience` | 大于 0 时启用早停 | 0 |
| `--optimizer` `--lr` `--weight-decay` | `sgd` `adamw` | `sgd` / 0.1 / 5e-4 |
| `--epochs` `--batch-size` `--seed` | | 60 / 128 / 42 |

名称以 `_pt` 结尾的是 ImageNet 预训练模型，其余从零训练。完整参数见 `python train.py --help`。

每次训练的产物保存在 `outputs/<实验名>/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次实验的完整参数 |
| `train.log` `history.json` | 每个 epoch 的损失、准确率、学习率 |
| `curves.png` | loss / accuracy / 学习率曲线 |
| `results.json` | 最优验证准确率、测试准确率、每类准确率、混淆矩阵、运行环境 |
| `confusion_matrix.png` `per_class_acc.png` `misclassified.png` | 测试集分析图表 |
| `best.pt` `last.pt` | 验证集最优权重、断点续训用的检查点（体积大，不入库） |

### 分组运行对照实验

```bash
python run_experiments.py --group arch
```

可选分组见下方"实验内容"。加 `--dry-run` 只打印每个实验对应的 `train.py` 命令而不执行；在 `--` 之后可以给每个实验追加参数，例如 `-- --workers 2`。

### 评估已训练的模型

```bash
python evaluate.py --run outputs/base_resnet18 --tta
```

输出总准确率、每类准确率、最易混淆的类别对，并在 `outputs/<实验名>/eval/` 下生成混淆矩阵和误分类样本图。`--tta` 表示对原图与水平翻转图的预测取平均。

### 可解释性分析

```bash
python explain.py --run outputs/base_resnet18
```

在 `outputs/<实验名>/explain/` 下生成三张图：分类正确样本的 Grad-CAM、误分类样本分别针对"预测类别"和"真实类别"的 Grad-CAM、各阶段的特征图。

## 实验内容

每组对照实验只改变一个因素，其余设置与基线 `base_resnet18` 保持一致（CIFAR 版 ResNet-18、basic 增强、SGD、cosine 调度、交叉熵、60 epoch、batch 128）。

| 分组 | 实验内容 |
|---|---|
| `arch` | 架构对比：自行设计的基线 CNN、ResNet-18、ResNet-18 + SE、ResNet-18 + CBAM |
| `hparam` | 基线 CNN 上学习率 × 权重衰减的网格搜索，用验证集准确率挑选 |
| `seeds` | 基线只换训练种子重复 3 次，估计随机波动的大小，作为判断其他对比是否显著的参照 |
| `pretrained` | ImageNet 预训练模型微调：ResNet-18、MobileNetV2、ViT-B/16 |
| `sched` | 学习率调度：恒定、阶梯衰减、余弦退火、warmup + 余弦、OneCycle、Plateau + 早停 |
| `aug` | 数据增强：无增强、裁剪 + 翻转、Cutout、AutoAugment、Mixup、CutMix |
| `loss` | 损失函数：交叉熵、标签平滑、Focal Loss |
| `final` | 最终模型：组合各组在验证集上表现最好的设置并延长训练 |

## 项目结构

```text
├── train.py               训练入口：训练、验证集选模型、测试集最终评估
├── evaluate.py            评估入口：准确率、混淆矩阵、误分类样本
├── explain.py             可解释性：Grad-CAM 与特征图
├── run_experiments.py     对照实验的配置与批量运行
├── compare.py             汇总各组实验的结果表格与对比曲线
├── src/
│   ├── data.py            数据加载、训练/验证分层划分、数据增强、Mixup / CutMix
│   ├── engine.py          单轮训练、评估、早停、优化器与学习率调度器
│   ├── losses.py          交叉熵（标签平滑）、Focal Loss
│   ├── gradcam.py         Grad-CAM
│   ├── visualize.py       训练曲线、混淆矩阵、每类准确率、误分类样本
│   ├── utils.py           随机种子、日志、运行环境记录
│   └── models/
│       ├── simple_cnn.py  自行设计的基线 CNN
│       ├── resnet.py      CIFAR 版 ResNet-18（可插入注意力模块）
│       ├── attention.py   SE、CBAM
│       └── pretrained.py  预训练模型微调封装
├── notebooks/
│   └── kaggle_run.ipynb   在 Kaggle GPU 上运行全部实验
├── outputs/               实验结果（日志、指标、图表；可由脚本重新生成）
└── requirements.txt
```

## 实现约定

- **数据划分**：官方 50,000 张训练图按类别分层划分为 45,000 张训练集和 5,000 张验证集（每类 500 张，划分种子固定为 42，与训练种子分开）。官方 10,000 张测试集不参与训练和调参，只在训练结束后用验证集上最优的权重评估一次。
- **模型选择**：每个 epoch 结束后在验证集上评估，保存验证准确率最高的权重；各组对照实验之间的取舍也只看验证集。
- **评估损失**：验证和测试损失统一使用不带标签平滑的交叉熵，使采用不同训练损失的实验可以直接比较。
- **训练准确率**：使用 Mixup / CutMix 时标签是两个类别的混合，训练准确率按混合系数加权统计，因此会明显低于验证准确率。
- **预训练模型**：输入在模型内部上采样到 224×224，归一化使用 ImageNet 的均值和方差；从零训练的模型直接使用 32×32 输入和 CIFAR-10 自身的统计量。
- **随机性**：固定 Python、NumPy、PyTorch 和 DataLoader 子进程的种子，并开启 cuDNN 确定性模式。断点续训不恢复随机数状态，续训结果与一次跑完的结果会有细微差别。
