"""预处理消融实验：严格只用训练部分（75/25 划分中的 5526 条），固定验证集不参与。

在训练部分内部做 5 折交叉验证（分层，且按讨论串分组），所有配置共用同一组折（配对比较），
每次只改一项，检验 PDF 列出的四种方法，并给出基线。

运行（需要 nltk 才能测词干化；没有 nltk 时会跳过这一项）：
    python experiments/preprocess_ablation.py
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess as P  # noqa: E402

N_FOLDS = 5
WORD = re.compile(r'(?u)\b\w\w+\b')               # sklearn 默认分词：丢掉标点
WORD_PUNCT = re.compile(r'(?u)\b\w\w+\b|[^\w\s]')  # 对照组：每个标点也作为一个词


def tokenize(doc, punct=False, stop=False, stem=None):
    toks = (WORD_PUNCT if punct else WORD).findall(doc.lower())
    if stop:
        toks = [w for w in toks if w not in ENGLISH_STOP_WORDS]   # 先去停用词，再词干化
    if stem:
        toks = [stem(w) for w in toks]
    return toks


def main():
    X, y, _ = P.load_data()
    train_idx, val_idx = P.get_split(X, y)
    X_tr, y_tr = [X[i] for i in train_idx], y[train_idx]          # 从这里开始只用训练部分
    assert not set(train_idx) & set(val_idx)

    raw = [P.clean_text(t, P.RAW_CFG) for t in X_tr]
    body = [P.clean_text(t, P.DEFAULT_CFG) for t in X_tr]
    configs = {
        'R  原始文本（不清洗）':    [tokenize(d) for d in raw],
        'B0 只用正文':            [tokenize(d) for d in body],
        'Bp 只用正文，保留标点':    [tokenize(d, punct=True) for d in body],
        'B1 只用正文 + 去停用词':   [tokenize(d, stop=True) for d in body],
    }
    try:
        from nltk.stem import PorterStemmer
        stem = lru_cache(maxsize=None)(PorterStemmer().stem)
        configs['B2 只用正文 + 去停用词 + 词干化'] = [tokenize(d, stop=True, stem=stem) for d in body]
    except ImportError:
        print('未安装 nltk，跳过词干化')

    models = {'SVM': lambda: LinearSVC(C=1.0, random_state=P.SEED),
              'NB': lambda: MultinomialNB(alpha=0.1)}
    groups = P.thread_ids(X)[train_idx]                              # 按讨论串分组分折
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=False).split(X_tr, y_tr, groups))
    acc = {}
    for name, docs in configs.items():
        for m, make in models.items():
            scores = []
            for fit_i, eval_i in folds:
                vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
                A = vec.fit_transform([docs[i] for i in fit_i])     # 只在本折的训练部分上 fit
                B = vec.transform([docs[i] for i in eval_i])
                scores.append((make().fit(A, y_tr[fit_i]).predict(B) == y_tr[eval_i]).mean() * 100)
            acc[name, m] = np.array(scores)

    print(f'训练部分 {len(y_tr)} 条，{N_FOLDS} 折分组交叉验证（固定验证集 {len(val_idx)} 条未使用）\n')
    print('各配置准确率（%，均值 ± 标准差）')
    for name in configs:
        print(f'  {name:26s}' + ''.join(f'{m}: {acc[name, m].mean():6.2f} ± {acc[name, m].std(ddof=1):.2f}    '
                                        for m in models))
    print('\n每种方法的效果（配对差值：用了该方法 − 没用）')
    names = list(configs)
    comparisons = [('去除正文以外的信息', names[1], names[0]),
                   ('去除标点', names[1], names[2]),
                   ('去除停用词', names[3], names[1])]
    if len(names) > 4:
        comparisons.append(('词干化', names[4], names[3]))
    for label, with_, without in comparisons:
        line = f'  {label:10s}'
        for m in models:
            d = acc[with_, m] - acc[without, m]
            line += f'{m}: {d.mean():+6.2f} ± {d.std(ddof=1):.2f}（{int((d > 0).sum())}/{N_FOLDS} 折更好）    '
        print(line)


if __name__ == '__main__':
    main()
