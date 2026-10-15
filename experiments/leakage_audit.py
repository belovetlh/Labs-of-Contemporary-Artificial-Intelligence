"""泄露审计：用程序核对整个流程中是否存在数据泄露。

只在训练部分上使用标签；验证集、测试集只检查文本和下标，不读标签。

    python experiments/leakage_audit.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import preprocess as P  # noqa: E402


def section(title):
    print(f'\n{"=" * 70}\n{title}\n{"=" * 70}')


X, y, X_test = P.load_data()
train_idx, val_idx = P.get_split(X, y)
X_tr = [P.clean_text(X[i]) for i in train_idx]
X_va = [P.clean_text(X[i]) for i in val_idx]
X_te = [P.clean_text(t) for t in X_test]
y_tr = y[train_idx]                                   # 只有训练部分的标签会被读取

section('[1] 划分')
print(f'训练部分 {len(train_idx)}，验证集 {len(val_idx)}，重叠 {len(set(train_idx) & set(val_idx))} 条，'
      f'合起来覆盖全部 {len(y)} 条：{len(set(train_idx) | set(val_idx)) == len(y)}')
saved = json.loads(P.split_path().read_text(encoding='utf-8'))
print(f'划分文件与当前使用的下标一致：{saved["train_idx"] == train_idx.tolist() and saved["val_idx"] == val_idx.tolist()}')
groups = P.thread_ids(X)
print(f'同时出现在训练部分和验证集的讨论串：{len(set(groups[train_idx]) & set(groups[val_idx]))} 个')


def norm(t):
    return re.sub(r'\s+', ' ', t).strip().lower()


tr_set = Counter(norm(t) for t in X_tr if norm(t))
va_norm = [norm(t) for t in X_va]
te_norm = [norm(t) for t in X_te]
dup_va = sum(1 for t in va_norm if t and t in tr_set)
dup_te = sum(1 for t in te_norm if t and (t in tr_set or t in set(va_norm)))
print(f'清洗后正文完全相同的文档：验证集中 {dup_va} 条也出现在训练部分（{dup_va / len(X_va):.2%}）；'
      f'测试集中 {dup_te} 条也出现在训练数据中（{dup_te / len(X_te):.2%}）')

section('[2] 清洗后文本中的残留信息')
patterns = {
    '头部字段行（From:/Subject:/Organization:/Lines:/NNTP...）':
        re.compile(r'^\s*(From|Subject|Organization|Lines|NNTP-Posting-Host|Reply-To|Newsgroups|Path|Message-ID)\s*:', re.M | re.I),
    '邮箱地址': re.compile(r'\S+@\S+\.\w+'),
    '引用行（> 开头）': re.compile(r'^\s*>', re.M),
    '引述行（xxx writes:）': re.compile(r'^.*(writes|wrote|says)\s*:\s*$', re.M | re.I),
}
for name, pat in patterns.items():
    counts = [sum(1 for t in part if pat.search(t)) for part in (X_tr, X_va, X_te)]
    print(f'  {name:45s} 训练部分 {counts[0]:4d}  验证集 {counts[1]:4d}  测试集 {counts[2]:4d}')

GROUP = re.compile(r'\b(comp|rec|sci|talk|alt|soc|misc|news)\.[a-z0-9][\w.+-]*', re.I)
hits = [(k, m.group(0).lower()) for k, t in enumerate(X_tr) for m in [GROUP.search(t)] if m]
print(f'\n  正文中提到新闻组名称（如 sci.crypt）：训练部分 {len(hits)} 条（{len(hits) / len(X_tr):.1%}）')
per_class = Counter(y_tr[k] for k, _ in hits)
print('  按类别：' + '，'.join(f'{P.CLASS_NAMES[c]} {per_class[c]}' for c in range(10)))
names = Counter(g for _, g in hits).most_common(8)
print('  最常见的名称：' + '，'.join(f'{g} {n}' for g, n in names))
by_name = {}
for k, g in hits:
    by_name.setdefault(g, Counter())[y_tr[k]] += 1
pure = sum(max(c.values()) for g, c in by_name.items() if sum(c.values()) >= 5)
total = sum(sum(c.values()) for g, c in by_name.items() if sum(c.values()) >= 5)
print(f'  出现 ≥5 次的名称中，帖子落在该名称"主导类别"的比例：{pure / max(total, 1):.0%}'
      '（越接近 100%，说明这类字符串越能直接指示类别）')

section('[3] TF-IDF 只在训练数据上 fit')
cfg = json.loads((P.OUTPUT_DIR / 'final_config.json').read_text(encoding='utf-8'))['tfidf']
cfg['ngram_range'] = tuple(cfg['ngram_range'])
vocab_tr = len(TfidfVectorizer(**cfg).fit(X_tr).vocabulary_)
vocab_trva = len(TfidfVectorizer(**cfg).fit(X_tr + X_va).vocabulary_)
reported = json.loads((P.OUTPUT_DIR / 'val_results.json').read_text(encoding='utf-8'))['vocab']
print(f'只用训练部分 fit 的词表：{vocab_tr:,}；训练部分 + 验证集 fit 的词表：{vocab_trva:,}；'
      f'阶段 2 实际使用的词表：{reported:,} → {"只用了训练部分 ✅" if reported == vocab_tr else "❌ 不一致"}')

section('[4] 代码中验证集 / 测试集的使用位置（静态检查）')
for fname in ('train.py', 'predict.py', 'experiments/preprocess_ablation.py', 'experiments/explore_train_only.py'):
    src = (ROOT / fname).read_text(encoding='utf-8').splitlines()
    uses = [(i + 1, l.strip()) for i, l in enumerate(src)
            if re.search(r"X_val|y_val|val_idx|X_test|X_te\b|test_docs|\bB\b.*=.*transform", l) and not l.strip().startswith('#')]
    print(f'\n  {fname}：')
    for ln, l in uses:
        print(f'    第 {ln:3d} 行：{l[:110]}')
