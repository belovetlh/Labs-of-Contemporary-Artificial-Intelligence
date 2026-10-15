"""实验一 第二部分：特征提取、模型训练与模型选择

流程（调参只用训练部分；验证集只用于比较各族的最佳配置；测试集留给 predict.py）：
  阶段 1  调参：在训练部分内做 5 折交叉验证（分层，且按讨论串分组）
          1a TF-IDF 设置 → 1b 逻辑回归 → 1c MLP → 1d 对照模型（朴素贝叶斯、线性 SVM）
          每一族用"一倍标准误规则"选配置：在交叉验证最高分一个标准误以内的配置里，选最简单的
  阶段 2  验证：各族最佳配置在整个训练部分上训练，在验证集上评估；
          逻辑回归与 MLP 之间用配对 bootstrap 置信区间决定最终模型；输出损失曲线

    python train.py              # 阶段 1 + 阶段 2
    python train.py --stage 2    # 只重跑阶段 2（读取阶段 1 保存的选择结果）

评价指标（实验前确定）：主要指标为准确率；辅助指标为 macro-F1、各类别 F1、混淆矩阵。
逻辑回归用不带隐藏层的 MLPClassifier 实现（softmax + 交叉熵），和 MLP 共用同一个训练循环。
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

import preprocess as P

SEED = P.SEED
N_FOLDS = 5
N_JOBS = 5              # 5 折并行
THREADS_PER_JOB = 2     # 每个进程的线程数固定，保证结果可复现
BATCH_SIZE = 128        # 目标批大小；实际把每轮均匀分成 ceil(n / 128) 批
PATIENCE = 3           # 留出折损失连续 3 轮不下降就停
N_BOOTSTRAP = 2000
CLASSES = np.arange(10)
OUT = P.OUTPUT_DIR
FIG_DIR = OUT / 'figures'
LOSS_DIR = OUT / 'loss_logs'

# ---------------------------------------------------------------- 事先确定的搜索范围（全部结果都报告）

# 去停用词：按讨论串分组的预处理消融中，两个模型都有超过波动的小幅提升（SVM +0.63±0.56，NB +0.38±0.36），
# 按事先定的规则采用（见 experiments/preprocess_ablation.py 与 outputs/preprocess_ablation_log.txt）
# 一倍标准误规则偏向"最简单"的一端，所以每个维度在"简单"方向上都要留出至少一档，
# 确保选中的配置落在范围内部而不是边缘（阶段 1 结束时会自动检查，见 check_edges）。
TFIDF_GRID = [dict(min_df=m, ngram_range=ng, sublinear_tf=s, stop_words='english')
              for m in (1, 2, 5, 10) for ng in ((1, 1), (1, 2)) for s in (False, True)]
# 种子 42 的首次运行中上限为 50 轮，学习率 0.003 和 0.01 的配置全部顶到上限，故提高到 150 轮
LR_GRID = [dict(hidden=(), alpha=a, lr=lr, max_epochs=150)
           for a in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0) for lr in (3e-3, 1e-2, 3e-2)]
# 6 种结构 × 4 种正则强度，所有配置统一 100 轮上限（公平比较：运行条件相同）。
MLP_GRID = [dict(hidden=h, alpha=a, lr=1e-3, max_epochs=100)
            for h in ((10,), (25,), (50,), (100,), (200,), (100, 50)) for a in (1e-4, 1e-3, 1e-2, 1e-1)]
NB_GRID = [dict(alpha=a) for a in (0.01, 0.1, 1.0, 10.0)]
SVM_GRID = [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)]
# 各族"更简单"的方向：值越大越简单（+1）或越小越简单（−1）；MLP 的结构按参数量判断
SIMPLER = {'tfidf': {'min_df': +1}, 'LR': {'alpha': +1}, 'MLP': {'alpha': +1},
           'NB': {'alpha': +1}, 'SVM': {'C': -1}}


# ---------------------------------------------------------------- 训练循环

def train_iterative(X_fit, y_fit, cfg, n_epochs, X_eval=None, y_eval=None, patience=None, track_train=False):
    """mini-batch 训练循环：每喂一个 batch 调用一次 partial_fit，记录这一步的损失。
    逻辑回归和 MLP 共用；cfg['hidden'] 为 () 时没有隐藏层，模型就是多分类逻辑回归。
    patience 不为 None 时按 X_eval 上的损失早停（只在交叉验证的留出折上使用）"""
    # batch_size='auto' 即 min(200, 传入样本数)：每次 partial_fit 传入的就是一个 batch（最多 BATCH_SIZE 条），
    # 所以正好整批做一次更新；每轮最后一批不满 BATCH_SIZE 条时也不会触发 sklearn 的截断警告
    clf = MLPClassifier(hidden_layer_sizes=cfg['hidden'], alpha=cfg['alpha'], learning_rate_init=cfg['lr'],
                        batch_size='auto', random_state=SEED)
    rng = np.random.RandomState(SEED)
    n = X_fit.shape[0]
    steps, history = [], []
    best_loss, wait = np.inf, 0
    n_batches = int(np.ceil(n / BATCH_SIZE))
    for epoch in range(1, n_epochs + 1):
        order = rng.permutation(n)
        # 均匀分批：每批大小最多差 1 条。若按"每批 128 条、余数单独成批"切，最后一批很小（5526 条时只有 22 条），
        # sklearn 的 L2 项按批大小归一化，这一步的正则会被放大约 5.8 倍，损失曲线每轮末尾出现尖刺
        for idx in np.array_split(order, n_batches):
            # sklearn 的 loss_ = 交叉熵 + 0.5·alpha·‖W‖²/批大小，正则项用更新前的权重计算。
            # 扣掉正则项得到纯交叉熵 batch_ce，与每轮在整个数据集上算的交叉熵口径一致；原始 loss_ 也照常保留
            penalty = l2_penalty(clf, cfg['alpha'], len(idx)) if hasattr(clf, 'coefs_') else None
            clf.partial_fit(X_fit[idx], y_fit[idx], classes=CLASSES)
            if penalty is None:          # 第一步之前权重还没初始化，用这一步更新后的权重近似
                penalty = l2_penalty(clf, cfg['alpha'], len(idx))
            steps.append({'step': len(steps) + 1, 'epoch': epoch, 'batch_loss': clf.loss_,
                          'batch_ce': clf.loss_ - penalty})
        rec = {'epoch': epoch}
        if track_train:
            proba = clf.predict_proba(X_fit)
            rec['train_loss'] = log_loss(y_fit, proba, labels=CLASSES)
            rec['train_acc'] = accuracy_score(y_fit, CLASSES[proba.argmax(1)])
        if X_eval is not None:
            proba = clf.predict_proba(X_eval)
            rec['eval_loss'] = log_loss(y_eval, proba, labels=CLASSES)
            rec['eval_acc'] = accuracy_score(y_eval, CLASSES[proba.argmax(1)])
        history.append(rec)
        if patience is not None:
            if rec['eval_loss'] < best_loss - 1e-4:
                best_loss, wait = rec['eval_loss'], 0
            else:
                wait += 1
                if wait >= patience:
                    break
    return clf, steps, history


def l2_penalty(clf, alpha, n_samples):
    """sklearn MLP 计入 loss_ 的 L2 正则项（只含权重，不含偏置）"""
    return 0.5 * alpha * sum(float((w * w).sum()) for w in clf.coefs_) / n_samples


def n_weights(vocab, hidden):
    sizes = [vocab, *hidden, len(CLASSES)]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def make_simple(kind, cfg):
    if kind == 'NB':
        return MultinomialNB(alpha=cfg['alpha'])
    if kind == 'SVM':
        return LinearSVC(C=cfg['C'], random_state=SEED)
    return LogisticRegression(max_iter=1000, random_state=SEED)   # 只用于筛选 TF-IDF 设置


# ---------------------------------------------------------------- 交叉验证中每一折的任务（在子进程里运行）

def _fold_tfidf(docs, y, fit_i, eval_i, params):
    with threadpool_limits(THREADS_PER_JOB):
        vec = TfidfVectorizer(**params)
        A = vec.fit_transform([docs[i] for i in fit_i])          # 只在本折的训练部分上 fit
        B = vec.transform([docs[i] for i in eval_i])
        model = make_simple('LR', {}).fit(A, y[fit_i])
    return {'acc': accuracy_score(y[eval_i], model.predict(B)), 'vocab': len(vec.vocabulary_)}


def _fold_iterative(A, yA, B, yB, cfg):
    with threadpool_limits(THREADS_PER_JOB):
        _, _, history = train_iterative(A, yA, cfg, cfg['max_epochs'], B, yB, patience=PATIENCE)
    h = pd.DataFrame(history)
    best = h.loc[h['eval_loss'].idxmin()]                         # 本折留出损失最低的那一轮
    return {'acc': float(best['eval_acc']), 'loss': float(best['eval_loss']),
            'best_epoch': int(best['epoch']), 'epochs_run': len(h)}


def _fold_simple(A, yA, B, yB, kind, cfg):
    with threadpool_limits(THREADS_PER_JOB):
        model = make_simple(kind, cfg).fit(A, yA)
    return {'acc': accuracy_score(yB, model.predict(B))}


# ---------------------------------------------------------------- 阶段 1

def vectorize_folds(docs, y, folds, params):
    """每一折单独 fit 一个 TF-IDF，只用该折的训练部分"""
    data = []
    for fit_i, eval_i in folds:
        vec = TfidfVectorizer(**params)
        data.append((vec.fit_transform([docs[i] for i in fit_i]), y[fit_i],
                     vec.transform([docs[i] for i in eval_i]), y[eval_i]))
    return data


def summarize(family, grid, fold_results, complexity):
    rows = []
    for cfg, res in zip(grid, fold_results):
        acc = np.array([r['acc'] for r in res])
        row = {'family': family, 'config': json.dumps(cfg), 'acc_mean': acc.mean(), 'acc_std': acc.std(ddof=1),
               'acc_se': acc.std(ddof=1) / np.sqrt(len(acc)), 'fold_acc': json.dumps(np.round(acc, 4).tolist()),
               'complexity': complexity(cfg, res)}
        if 'best_epoch' in res[0]:
            row['best_epoch'] = int(np.median([r['best_epoch'] for r in res]))
            row['fold_best_epochs'] = json.dumps([r['best_epoch'] for r in res])
        if 'vocab' in res[0]:
            row['vocab'] = int(np.mean([r['vocab'] for r in res]))
        rows.append(row)
    df = pd.DataFrame(rows)
    best = df.loc[df['acc_mean'].idxmax()]
    tied = df[df['acc_mean'] >= best['acc_mean'] - best['acc_se']]            # 一倍标准误规则
    chosen = tied.sort_values('complexity', kind='stable').index[0]
    df['within_1se'] = df.index.isin(tied.index)
    df['selected'] = df.index == chosen
    return df


def print_family(df):
    for _, r in df.sort_values('acc_mean', ascending=False).iterrows():
        mark = '★' if r['selected'] else ('·' if r['within_1se'] else ' ')
        extra = f"  最佳轮数 {r['best_epoch']}" if 'best_epoch' in r and pd.notna(r.get('best_epoch')) else ''
        print(f"   {mark} {r['acc_mean'] * 100:6.2f} ± {r['acc_std'] * 100:.2f}  {r['config']}{extra}")
    print('   （★ 选中；· 在最高分一个标准误以内）')


def check_edges(family, df):
    """检查选中的配置是否落在搜索范围的边缘。一倍标准误规则偏向简单的一端，落在"简单"端的边缘说明范围可能不够"""
    grid = [json.loads(c) for c in df['config']]
    chosen = json.loads(df.loc[df['selected'], 'config'].iloc[0])
    notes = []
    for key, direction in SIMPLER[family].items():
        values = sorted({g[key] for g in grid})
        if chosen[key] == values[-1 if direction > 0 else 0]:
            notes.append(f'{key}={chosen[key]} 在"简单"一端的边缘')
        elif chosen[key] == values[0 if direction > 0 else -1]:
            notes.append(f'{key}={chosen[key]} 在"复杂"一端的边缘')
    if family == 'MLP':
        single = sorted({g['hidden'][0] for g in grid if len(g['hidden']) == 1})
        if len(chosen['hidden']) == 1 and chosen['hidden'][0] == single[0]:
            notes.append(f'hidden={chosen["hidden"]} 是最小的结构')
    print('   边界检查：' + ('⚠️ ' + '；'.join(notes) if notes else '✓ 选中的配置在搜索范围内部'))
    return notes


def stage1(docs, y, groups):
    # 按讨论串分组分折：同一讨论串的帖子不会同时出现在某一折的训练部分和留出部分。
    # 不打乱：打乱会破坏 StratifiedGroupKFold 的分层（实测类别占比偏离可达 10 个点，不打乱 ≤ 0.2）
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=False).split(docs, y, groups))
    par = Parallel(n_jobs=N_JOBS)
    summaries = []

    t = time.time()
    print('\n[1a] TF-IDF 设置（用 sklearn 逻辑回归快速筛选；词表越小越简单）')
    res = [par(delayed(_fold_tfidf)(docs, y, a, b, p) for a, b in folds) for p in TFIDF_GRID]
    grid = [{**p, 'ngram_range': list(p['ngram_range'])} for p in TFIDF_GRID]
    df = summarize('tfidf', grid, res, lambda cfg, r: np.mean([x['vocab'] for x in r]))
    print_family(df)
    check_edges('tfidf', df)
    summaries.append(df)
    tfidf = json.loads(df.loc[df['selected'], 'config'].iloc[0])
    tfidf['ngram_range'] = tuple(tfidf['ngram_range'])
    print(f'   用时 {time.time() - t:.0f} 秒')

    fold_data = vectorize_folds(docs, y, folds, tfidf)
    vocab = int(np.mean([d[0].shape[1] for d in fold_data]))

    for family, grid, complexity in [
        ('LR', LR_GRID, lambda c, r: (-c['alpha'], np.median([x['best_epoch'] for x in r]))),
        ('MLP', MLP_GRID, lambda c, r: (n_weights(vocab, c['hidden']), -c['alpha'])),
    ]:
        t = time.time()
        print(f'\n[1{"b" if family == "LR" else "c"}] {family}（{len(grid)} 组 × {N_FOLDS} 折，早停看留出折损失）')
        res = [par(delayed(_fold_iterative)(*d, cfg) for d in fold_data) for cfg in grid]
        grid_json = [{**c, 'hidden': list(c['hidden'])} for c in grid]
        df = summarize(family, grid_json, res, lambda c, r, f=complexity: f({**c, 'hidden': tuple(c['hidden'])}, r))
        print_family(df)
        check_edges(family, df)
        summaries.append(df)
        print(f'   用时 {time.time() - t:.0f} 秒')

    t = time.time()
    print('\n[1d] 对照模型')
    for family, grid, complexity in [('NB', NB_GRID, lambda c, r: -c['alpha']),
                                     ('SVM', SVM_GRID, lambda c, r: c['C'])]:
        res = [par(delayed(_fold_simple)(*d, family, cfg) for d in fold_data) for cfg in grid]
        df = summarize(family, grid, res, complexity)
        print(f'  {family}:')
        print_family(df)
        check_edges(family, df)
        summaries.append(df)
    print(f'   用时 {time.time() - t:.0f} 秒')

    all_df = pd.concat(summaries, ignore_index=True)
    all_df['complexity'] = all_df['complexity'].astype(str)
    all_df.to_csv(OUT / 'cv_results.csv', index=False, encoding='utf-8-sig')
    selection = {'tfidf': {**tfidf, 'ngram_range': list(tfidf['ngram_range'])}}
    for family in ('LR', 'MLP', 'NB', 'SVM'):
        row = all_df[(all_df['family'] == family) & all_df['selected']].iloc[0]
        selection[family] = {'config': json.loads(row['config']), 'cv_acc': row['acc_mean'], 'cv_std': row['acc_std']}
        if family in ('LR', 'MLP'):
            selection[family]['epochs'] = int(row['best_epoch'])
    (OUT / 'cv_selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    return selection


# ---------------------------------------------------------------- 阶段 2

def val_metrics(y_true, y_pred):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'per_class_f1': np.round(f1_score(y_true, y_pred, average=None), 4).tolist(),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()}


def paired_bootstrap(y_true, pred_a, pred_b, groups):
    """验证集上 准确率(b) − 准确率(a) 的配对 bootstrap 95% 置信区间。
    按讨论串整组重抽样：同一讨论串的帖子彼此相关，按单个帖子抽样会低估不确定性"""
    rng = np.random.RandomState(SEED)
    ca, cb = (pred_a == y_true).astype(float), (pred_b == y_true).astype(float)
    _, g = np.unique(groups, return_inverse=True)
    n_groups = g.max() + 1
    sum_a, sum_b, size = (np.bincount(g, weights=w, minlength=n_groups) for w in (ca, cb, np.ones_like(ca)))
    diffs = []
    for _ in range(N_BOOTSTRAP):
        pick = np.bincount(rng.randint(0, n_groups, n_groups), minlength=n_groups)   # 每个讨论串被抽中的次数
        diffs.append((pick @ sum_b - pick @ sum_a) / (pick @ size))
    return float(cb.mean() - ca.mean()), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


MODEL_NAMES = {'LR': '逻辑回归', 'MLP': 'MLP'}


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    return plt


def plot_loss(rows, path, n_train, n_val):
    """rows 为 [(family, steps, history), ...]，每个模型一行三张子图：
    每一步的 mini-batch 交叉熵 → 每轮平均交叉熵 → 每轮准确率。所有曲线都直接用记录的原始数值绘制"""
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.5 * len(rows) + 0.4), squeeze=False)
    for (family, steps, history), (ax1, ax2, ax3) in zip(rows, axes):
        s, h, name = pd.DataFrame(steps), pd.DataFrame(history), MODEL_NAMES[family]
        ax1.plot(s['step'], s['batch_ce'], lw=0.6)
        for e in h['epoch'][:-1]:
            ax1.axvline(s.loc[s['epoch'] == e, 'step'].max() + 0.5, color='grey', lw=0.3, alpha=0.35)
        ax1.set_yscale('log')
        ax1.set(xlabel='训练步（每步一个 mini-batch）', ylabel='交叉熵（对数坐标）', title=f'{name}：每一步的训练损失')
        ax2.plot(h['epoch'], h['train_loss'], 'o-', ms=3, label='训练部分')
        ax2.plot(h['epoch'], h['eval_loss'], 's-', ms=3, label='验证集')
        ax2.set_yscale('log')
        ax2.set(xlabel='epoch', ylabel='交叉熵（对数坐标）', title=f'{name}：每轮平均损失')
        ax3.plot(h['epoch'], h['train_acc'] * 100, 'o--', ms=3, label='训练部分')
        ax3.plot(h['epoch'], h['eval_acc'] * 100, 's-', ms=3, label='验证集')
        ax3.set(xlabel='epoch', ylabel='准确率（%）', title=f'{name}：每轮准确率')
        for ax in (ax2, ax3):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
    fig.text(0.5, 0.008, '左：每个 mini-batch（约 128 条）上的交叉熵，不含 L2 正则项，原始数据未平滑，灰线为 epoch 分界。'
             f'中、右：每轮结束时在整个训练部分（{n_train} 条）和验证集（{n_val} 条）上计算的平均值。'
             '验证集只用于展示，不参与训练和早停。', ha='center', fontsize=8.5, color='dimgray')
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(name, cm):
    plt = _pyplot()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(10), P.CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(10), P.CLASS_NAMES, fontsize=8)
    ax.set(xlabel='预测类别', ylabel='真实类别', title=f'{name}：验证集混淆矩阵')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'confusion_{name}.png', dpi=150)
    plt.close(fig)


def stage2(data, selection):
    tfidf = {**selection['tfidf'], 'ngram_range': tuple(selection['tfidf']['ngram_range'])}
    vec = TfidfVectorizer(**tfidf)
    A = vec.fit_transform(data['X_train'])                           # 只在训练部分上 fit
    B = vec.transform(data['X_val'])
    yA, yB = data['y_train'], data['y_val']
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    results, preds, curves = {}, {}, []

    for family in ('LR', 'MLP'):
        cfg = {**selection[family]['config'], 'hidden': tuple(selection[family]['config']['hidden'])}
        epochs = selection[family]['epochs']
        t = time.time()
        # 训练轮数固定为交叉验证选出的轮数；验证集损失只记录下来画图，不用于任何决定
        clf, steps, history = train_iterative(A, yA, cfg, epochs, B, yB, track_train=True)
        pd.DataFrame(steps).to_csv(LOSS_DIR / f'{family}_steps.csv', index=False)
        pd.DataFrame(history).to_csv(LOSS_DIR / f'{family}_epochs.csv', index=False)
        preds[family] = clf.predict(B)
        results[family] = {'config': selection[family]['config'], 'epochs': epochs, **val_metrics(yB, preds[family]),
                           'train_seconds': round(time.time() - t, 1)}
        plot_loss([(family, steps, history)], FIG_DIR / f'loss_{family}.png', len(yA), len(yB))
        curves.append((family, steps, history))
    plot_loss(curves, FIG_DIR / 'loss_curves.png', len(yA), len(yB))   # 两个模型合在一张图里，供报告使用

    for family in ('NB', 'SVM'):
        model = make_simple(family, selection[family]['config']).fit(A, yA)
        preds[family] = model.predict(B)
        results[family] = {'config': selection[family]['config'], **val_metrics(yB, preds[family])}

    diff, ci = paired_bootstrap(yB, preds['LR'], preds['MLP'], data['groups_val'])
    if ci[0] > 0:
        final, reason = 'MLP', 'MLP 显著优于逻辑回归（置信区间在 0 以上）'
    elif ci[1] < 0:
        final, reason = 'LR', '逻辑回归显著优于 MLP（置信区间在 0 以下）'
    else:
        final, reason = 'LR', '两者差距不显著（置信区间包含 0），选更简单的逻辑回归'
    plot_confusion(final, results[final]['confusion_matrix'])

    decision = {'mlp_minus_lr': diff, 'ci95': ci, 'final_model': final, 'reason': reason}
    report = {'tfidf': selection['tfidf'], 'vocab': len(vec.vocabulary_), 'models': results, 'decision': decision}
    (OUT / 'val_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'final_config.json').write_text(json.dumps({
        'tfidf': selection['tfidf'], 'family': final, 'config': selection[final]['config'],
        'epochs': selection[final].get('epochs')}, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\n[2] 验证集比较（各族最佳配置在训练部分上训练，在验证集上评估）  词表 {len(vec.vocabulary_):,}')
    for family in ('LR', 'MLP', 'NB', 'SVM'):
        r = results[family]
        cv = selection[family]
        print(f"   {family:4s} 准确率 {r['accuracy'] * 100:6.2f}  macro-F1 {r['macro_f1'] * 100:6.2f}"
              f"   （交叉验证 {cv['cv_acc'] * 100:.2f} ± {cv['cv_std'] * 100:.2f}）  {json.dumps(r['config'])}")
    print(f'   MLP − 逻辑回归：{diff * 100:+.2f} 个点，95% 置信区间 [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]')
    print(f'   最终模型：{final}（{reason}）')


# ---------------------------------------------------------------- 入口

def run_info():
    import scipy
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'seed': SEED, 'n_folds': N_FOLDS, 'n_jobs': N_JOBS, 'threads_per_job': THREADS_PER_JOB,
            'batch_size': BATCH_SIZE, 'patience': PATIENCE, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', '2'], default='all')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / 'run_info.json').write_text(json.dumps(run_info(), indent=2), encoding='utf-8')

    data = P.prepare()
    assert not set(data['train_idx']) & set(data['val_idx'])
    print(f"训练部分 {len(data['y_train'])} 条，验证集 {len(data['y_val'])} 条")
    t0 = time.time()
    if args.stage == 'all':
        selection = stage1(data['X_train'], data['y_train'], data['groups_train'])   # 阶段 1 只用训练部分
    else:
        selection = json.loads((OUT / 'cv_selection.json').read_text(encoding='utf-8'))
    stage2(data, selection)
    print(f'\n总用时 {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
