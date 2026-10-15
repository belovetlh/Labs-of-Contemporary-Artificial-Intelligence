"""生成报告用的图（只读取已有的结果文件，不训练模型，不使用测试集）。

    python experiments/report_figures.py

输出到 outputs/figures/report/：
    fig1_loss.png            训练结果：最终模型（逻辑回归）的逐步损失、每轮损失、每轮准确率
    fig1_loss_both.png       同上，另加 MLP 一行作对照
    fig2_params.png          参数对比：逻辑回归的正则强度、MLP 的网络大小（交叉验证）
    fig3_confusion.png       分类性能：最终模型在验证集上的混淆矩阵（按行归一化）
    fig4_split.png           随机划分 vs 按讨论串分组划分：四个模型的验证准确率
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocess as P  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.0})
OUT = P.OUTPUT_DIR
FIG = OUT / 'figures' / 'report'
PAGE_W = 6.3                                    # A4 版心宽度约 16 cm
NAMES = {'LR': '逻辑回归', 'MLP': 'MLP', 'NB': '朴素贝叶斯', 'SVM': '线性 SVM'}
C_TRAIN, C_VAL = '#1f77b4', '#ff7f0e'


def set_loss_scale(ax, values):
    """损失跨越超过 5 倍时用对数坐标，刻度写成普通数字"""
    if values.max() / values.min() > 5:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        return '交叉熵（对数坐标）'
    return '交叉熵'


def loss_row(axes, family):
    s = pd.read_csv(OUT / 'loss_logs' / f'{family}_steps.csv')
    h = pd.read_csv(OUT / 'loss_logs' / f'{family}_epochs.csv')
    name = NAMES[family]
    a1, a2, a3 = axes
    a1.plot(s['step'], s['batch_ce'], lw=0.5, color=C_TRAIN)
    a1.set(xlabel='训练步（mini-batch）', ylabel=set_loss_scale(a1, s['batch_ce']), title=f'{name}：每一步的训练损失')
    a2.plot(h['epoch'], h['train_loss'], 'o-', ms=2, color=C_TRAIN, label='训练部分')
    a2.plot(h['epoch'], h['eval_loss'], 's-', ms=2, color=C_VAL, label='验证集')
    a2.set(xlabel='epoch', ylabel=set_loss_scale(a2, pd.concat([h['train_loss'], h['eval_loss']])), title=f'{name}：每轮平均损失')
    a3.plot(h['epoch'], h['train_acc'] * 100, 'o--', ms=2, color=C_TRAIN, label='训练部分')
    a3.plot(h['epoch'], h['eval_acc'] * 100, 's-', ms=2, color=C_VAL, label='验证集')
    a3.set(xlabel='epoch', ylabel='准确率（%）', title=f'{name}：每轮准确率')
    for ax in (a2, a3):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)


def fig1():
    note = ('左：每个 mini-batch（约 128 条）上的交叉熵，不含 L2 正则项，原始数据未平滑。\n'
            '中、右：每轮结束时在整个训练部分和验证集上的平均值；验证集只用于展示，不参与训练和早停。')
    for families, name in [(['LR'], 'fig1_loss.png'), (['LR', 'MLP'], 'fig1_loss_both.png')]:
        fig, axes = plt.subplots(len(families), 3, figsize=(PAGE_W, 2.0 * len(families) + 0.45), squeeze=False)
        for fam, row in zip(families, axes):
            loss_row(row, fam)
        fig.text(0.5, 0.005, note, ha='center', fontsize=6.5, color='dimgray')
        fig.tight_layout(rect=(0, 0.09 if len(families) == 1 else 0.05, 1, 1), w_pad=1.2)
        fig.savefig(FIG / name, dpi=300)
        plt.close(fig)


def fig2():
    cv = pd.read_csv(OUT / 'cv_results.csv')
    cv['cfg'] = cv['config'].map(json.loads)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 3.0))

    lr = cv[cv['family'] == 'LR'].copy()
    lr['alpha'] = lr['cfg'].map(lambda c: c['alpha'])
    lr['lr'] = lr['cfg'].map(lambda c: c['lr'])
    best = lr.loc[lr['acc_mean'].idxmax()]
    thr = (best['acc_mean'] - best['acc_se']) * 100
    a1.axhspan(thr, best['acc_mean'] * 100, color='gray', alpha=0.12, lw=0, label='最高分一个标准误以内')
    for (rate, d), mk in zip(lr.groupby('lr'), 'o^s'):
        d = d.sort_values('alpha')
        a1.plot(d['alpha'], d['acc_mean'] * 100, marker=mk, ms=3, label=f'学习率 {rate:g}')
    sel = lr[lr['selected']].iloc[0]
    a1.plot(sel['alpha'], sel['acc_mean'] * 100, '*', ms=10, color='crimson', zorder=5, label='规则选中')
    a1.set_xscale('log')
    a1.set_ylim(72, 79.5)
    a1.set(xlabel='L2 正则强度 alpha（对数）', ylabel='交叉验证准确率（%）', title='逻辑回归：正则强度')

    mlp = cv[cv['family'] == 'MLP'].copy()
    mlp['alpha'] = mlp['cfg'].map(lambda c: c['alpha'])
    mlp['hidden'] = mlp['cfg'].map(lambda c: tuple(c['hidden']))
    best = mlp.loc[mlp['acc_mean'].idxmax()]
    thr = (best['acc_mean'] - best['acc_se']) * 100
    a2.axhspan(thr, best['acc_mean'] * 100, color='gray', alpha=0.12, lw=0)
    one = mlp[mlp['hidden'].map(len) == 1].copy()
    one['size'] = one['hidden'].map(lambda h: h[0])
    for (alpha, d), mk in zip(one.groupby('alpha'), 'o^sD'):
        d = d.sort_values('size')
        line = a2.plot(d['size'], d['acc_mean'] * 100, marker=mk, ms=3, label=f'alpha={alpha:g}')[0]
        two = mlp[(mlp['hidden'] == (100, 50)) & (mlp['alpha'] == alpha)]
        a2.plot([100] * len(two), two['acc_mean'] * 100, marker=mk, ms=4, ls='none', mfc='white',
                color=line.get_color())
    sel = mlp[mlp['selected']].iloc[0]
    a2.plot(sel['hidden'][0], sel['acc_mean'] * 100, '*', ms=10, color='crimson', zorder=5)
    a2.plot([], [], 'o', ms=4, mfc='white', color='gray', ls='none', label='两层 (100, 50)')
    a2.set_xscale('log')
    a2.set_xticks([10, 25, 50, 100, 200], ['10', '25', '50', '100', '200'])
    a2.set_ylim(75.5, 79.5)
    a2.set(xlabel='隐藏层神经元数（对数）', ylabel='交叉验证准确率（%）', title='MLP：网络大小')
    for ax in (a1, a2):
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5,
                  columnspacing=1.0, handlelength=1.6)
    fig.text(0.5, 0.005, '训练部分内 5 折交叉验证（按讨论串分组）的平均准确率。灰色带：最高分一个标准误以内；★：一倍标准误规则选中的配置。\n'
             '左图中学习率 0.03、alpha=1 的一点为 65.2，超出纵轴范围；右图空心点为两层结构 (100, 50)。',
             ha='center', fontsize=6.5, color='dimgray')
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
    fig.savefig(FIG / 'fig2_params.png', dpi=300)
    plt.close(fig)


def fig3():
    val = json.loads((OUT / 'val_results.json').read_text(encoding='utf-8'))
    family = val['decision']['final_model']
    cm = np.array(val['models'][family]['confusion_matrix'], dtype=float)
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f'{pct[i, j]:.0f}', ha='center', va='center', fontsize=6,
                        color='white' if pct[i, j] > 50 else 'black')
    ax.set_xticks(range(10), P.CLASS_NAMES, rotation=50, ha='right', fontsize=6.5)
    ax.set_yticks(range(10), P.CLASS_NAMES, fontsize=6.5)
    acc = val['models'][family]['accuracy'] * 100
    ax.set(xlabel='预测类别', ylabel='真实类别', title=f'{NAMES[family]}：验证集混淆矩阵（%，按行归一化）\n准确率 {acc:.2f}%')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_confusion.png', dpi=300)
    plt.close(fig)


def fig4():
    # 早期随机划分时的验证集结果（与当前验证集不同，只用于对比）
    rnd = json.loads((OUT / 'random_split_val_results.json').read_text(encoding='utf-8'))['models']
    grp = json.loads((OUT / 'val_results.json').read_text(encoding='utf-8'))['models']
    fams = ['LR', 'MLP', 'NB', 'SVM']
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    b1 = ax.bar(x - 0.19, [rnd[f]['accuracy'] * 100 for f in fams], 0.38, label='随机划分', color='#bbbbbb')
    b2 = ax.bar(x + 0.19, [grp[f]['accuracy'] * 100 for f in fams], 0.38, label='按讨论串分组', color=C_TRAIN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f'{b.get_height():.1f}',
                    ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x, [NAMES[f] for f in fams])
    ax.set_ylim(76, 86)
    ax.set(ylabel='验证准确率（%）', title='两种划分方式下的验证准确率')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    fig.text(0.5, 0.005, '两种划分的验证集不同；各模型为各自流程中按同一规则选出的最佳配置。', ha='center', fontsize=6, color='dimgray')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / 'fig4_split.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()
    print('\n'.join(sorted(p.name for p in FIG.glob('*.png'))))
