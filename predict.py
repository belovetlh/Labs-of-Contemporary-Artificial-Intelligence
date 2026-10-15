"""实验一 最后一步：用最终模型对测试集预测一次，并按助教样例核对提交格式。

提交的模型就是验证过的模型：只用训练部分训练，配置、训练轮数、种子都与 train.py 阶段 2 相同。
验证集不参与拟合（PDF 第 3 页："验证集……不能参与模型拟合"）。
为证明是同一个模型，会重新计算它在验证集上的准确率，必须与 outputs/val_results.json 中的记录完全一致。

需要先运行 train.py。

    python predict.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import preprocess as P
from train import LOSS_DIR, OUT, make_simple, train_iterative

SUBMISSION = OUT / 'predictions_final.csv'
SAMPLE = P.DATA_DIR / 'predictions.csv'          # 助教给的格式样例，只读不写


def check_format(path, n_test):
    """按样例核对：行数与测试集一致、单列、没有表头、取值在 0–9 之间"""
    sample = pd.read_csv(SAMPLE, header=None)
    ours = pd.read_csv(path, header=None)
    first_line = path.read_text(encoding='utf-8').splitlines()[0]
    checks = {
        f'行数 = 测试集条数 {n_test}': len(ours) == n_test,
        f'行数 = 样例行数 {len(sample)}': len(ours) == len(sample),
        '单列，与样例一致': ours.shape[1] == sample.shape[1] == 1,
        '没有表头（第一行是数字）': first_line.strip().isdigit(),
        '取值都在 0–9 之间': ours[0].between(0, 9).all(),
    }
    for name, ok in checks.items():
        print(f'   {"✅" if ok else "❌"} {name}')
    return all(checks.values())


def main():
    final = json.loads((OUT / 'final_config.json').read_text(encoding='utf-8'))
    recorded = json.loads((OUT / 'val_results.json').read_text(encoding='utf-8'))
    family = final['family']
    print(f'最终模型：{family}  {json.dumps(final["config"])}  训练轮数 {final.get("epochs")}  种子 {P.SEED}')

    data = P.prepare()
    tfidf = {**final['tfidf'], 'ngram_range': tuple(final['tfidf']['ngram_range'])}
    vec = TfidfVectorizer(**tfidf)
    A = vec.fit_transform(data['X_train'])            # 只用训练部分，与阶段 2 相同
    B = vec.transform(data['X_val'])
    T = vec.transform(data['X_test'])
    print(f'训练部分 {A.shape[0]} 条，测试集 {T.shape[0]} 条，词表 {A.shape[1]:,}')

    t = time.time()
    if family in ('LR', 'MLP'):
        cfg = {**final['config'], 'hidden': tuple(final['config']['hidden'])}
        model, steps, _ = train_iterative(A, data['y_train'], cfg, final['epochs'])
        LOSS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(steps).to_csv(LOSS_DIR / f'final_{family}_steps.csv', index=False)
    else:
        model = make_simple(family, final['config']).fit(A, data['y_train'])
    print(f'训练用时 {time.time() - t:.1f} 秒')

    val_acc = float((model.predict(B) == data['y_val']).mean())
    same = val_acc == recorded['models'][family]['accuracy']
    print(f'\n与验证过的模型是否相同：验证准确率 {val_acc * 100:.2f}，记录值 '
          f'{recorded["models"][family]["accuracy"] * 100:.2f} → {"✅ 完全一致" if same else "❌ 不一致，停止"}')
    if not same:
        raise SystemExit('模型与 train.py 阶段 2 不一致，没有生成提交文件')

    pred = model.predict(T)                           # 测试集只预测这一次
    pd.DataFrame(pred.astype(int)).to_csv(SUBMISSION, index=False, header=False)
    print(f'\n预测结果已写入 {SUBMISSION}')
    counts = np.bincount(pred, minlength=10)
    print('各类别预测条数：' + '，'.join(f'{P.CLASS_NAMES[c]} {counts[c]}' for c in range(10)))

    print('\n格式核对：')
    ok = check_format(SUBMISSION, len(data['X_test']))
    print('全部通过，可以提交' if ok else '有检查未通过，请先修正')


if __name__ == '__main__':
    main()
