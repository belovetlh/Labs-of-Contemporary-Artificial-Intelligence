"""数据摸底（规范流程的第二步）：验证集划分好之后，只用训练部分（5526 条）分析数据、为清洗规则提供依据。
固定验证集全程不参与。

    python experiments/explore_train_only.py

各节对应的决策：
    [1] 头部泄露强度            → 删除整个头部
    [2] 标题与讨论串            → 不用标题（用户决定）；随机划分 vs 分组划分的差距，说明为什么要分组划分
    [3] 驱动每条规则的现象      → 各条规则的具体写法
    [4] 数字的类别区分度        → 保留数字
    [5] 清洗前后的高频词        → 确认噪声被清掉
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocess as P  # noqa: E402

X, y, _ = P.load_data()                            # 测试集不参与任何分析
train_idx, val_idx = P.get_split(X, y)
T = [X[i] for i in train_idx]                     # 从这里开始只用训练部分
Y = y[train_idx]
ROW = list(train_idx)                              # 训练部分在原始数据中的行号
in_train = set(ROW)
thread = P.thread_ids(X)[train_idx]                # 讨论串编号
folds = list(StratifiedGroupKFold(5, shuffle=False).split(T, Y, thread))   # 主分析：按讨论串分组
random_folds = list(StratifiedKFold(5, shuffle=True, random_state=P.SEED).split(T, Y))       # 只用于 [2] 的对照


def body_of(t):
    return P.split_header(t)[1]


def cv_accuracy(docs, splits, make_vec, make_clf):
    accs = []
    for a, b in splits:
        vec = make_vec()
        clf = make_clf().fit(vec.fit_transform([docs[i] for i in a]), Y[a])
        accs.append(clf.score(vec.transform([docs[i] for i in b]), Y[b]) * 100)
    return np.array(accs)


print(f'只用训练部分 {len(T)} 条，5 折分组交叉验证（验证集 {len(val_idx)} 条不参与，测试集不参与）')

# ---------------------------------------------------------------- [1]
print('\n[1] 头部泄露强度：只用 Organization + 发件人邮箱域名做分类')
def meta(t):
    h = P.split_header(t)[0]
    m = re.search(r'^Organization:\s*(.*)$', h, re.M)
    o = m.group(1).strip().lower() if m else ''
    m = re.search(r'^From:.*?@([\w.-]+)', h, re.M)
    return o + ' | ' + (m.group(1).lower() if m else '')
acc = cv_accuracy([meta(t) for t in T], folds,
                  lambda: TfidfVectorizer(token_pattern=r'[\w.-]+'),
                  lambda: LogisticRegression(max_iter=2000))
print(f'    准确率 {acc.mean():.1f}% ± {acc.std(ddof=1):.1f}（随机猜 10%）')

# ---------------------------------------------------------------- [2]
print('\n[2] 标题与讨论串')
def subject(t):
    m = P.SUBJECT.search(P.split_header(t)[0])
    return m.group(1) if m else ''
raw_subj = [subject(t) for t in T]
print(f'    标题以 Re:/Fwd: 开头（从原帖继承）：{np.mean([bool(P.RE_PREFIX.match(s)) for s in raw_subj]):.1%}')
body = [P.clean_text(t) for t in T]
subj_only = [P.RE_PREFIX.sub('', s).strip() for s in raw_subj]
with_subj = [P.clean_text(t, P.CleanConfig(keep_subject=True)) for t in T]
svm = lambda: LinearSVC(C=1.0, random_state=P.SEED)
tfidf = lambda: TfidfVectorizer(min_df=2, sublinear_tf=True)
print(f'    {"输入":10s}{"随机划分（讨论串重叠）":>22s}{"按讨论串分组（全新讨论串）":>26s}')
res = {}
for name, docs in [('只用标题', subj_only), ('只用正文', body), ('正文+标题', with_subj)]:
    r = cv_accuracy(docs, random_folds, tfidf, svm)
    g = cv_accuracy(docs, folds, tfidf, svm)
    res[name] = (r, g)
    print(f'    {name:10s}{r.mean():>14.2f} ± {r.std(ddof=1):.2f}{g.mean():>18.2f} ± {g.std(ddof=1):.2f}')
for i, split_name in enumerate(['随机划分', '分组划分']):
    d = res['正文+标题'][i] - res['只用正文'][i]
    print(f'    加标题的提升（{split_name}）：{d.mean():+.2f} ± {d.std(ddof=1):.2f}')

# ---------------------------------------------------------------- [3]
print('\n[3] 驱动每条规则的现象，在训练部分内部是否存在')
examples = {1740: '分隔行含空格 → 头部按"只含空白的行"切分', 920: '横线分隔线后是正文 → 签名只认标准 --',
            3911: '| 开头的矩阵 → 单独的 | 不算引用', 2436: '棒球积分榜 → 乱码规则只删 uuencode',
            71: '正文开头的内层头部 → 一并剥离', 3642: '编码块后的 -- 被误删 → 尾行长度校验'}
for row, why in examples.items():
    print(f'    典型样本 row {row:5d} 在{"训练部分" if row in in_train else "验证集"}：{why}')

sep_wrong = sum(1 for t in T if t.find('\n\n') != P.HEADER_SEP.search(t).start())
print(f'    分隔行含空白、按严格空行切分会切错位置：{sep_wrong} 个帖子')

dash_body = 0
for t in T:
    lines = [l for l in body_of(t).rstrip().split('\n') if not P.QUOTE.match(l)]
    for k in range(len(lines) - 1, -1, -1):
        if re.fullmatch(r'\s*--+\s*', lines[k]):
            after = lines[k + 1:]
            if (not P.SIG_SEP.match(lines[k]) and len(after) <= 15
                    and sum(1 for l in after if len(l.split()) >= 8) >= 2):
                dash_body += 1
            break
print(f'    长横线分隔线之后跟着成段文字（任意横线都当签名会误删）：{dash_body} 个帖子')

pipe_table = sum(1 for t in T if any(re.match(r'^[ \t]*\|', l) and not P.QUOTE.match(l) and re.search(r'\d.*\|', l)
                                     for l in body_of(t).split('\n')))
print(f'    单独 | 开头、含数字的表格行（把 | 当引用会误删）：{pipe_table} 个帖子')

def low_alpha(l):
    s = l.strip()
    return len(s) >= 40 and sum(c.isalpha() for c in s) / len(s) < 0.5
old_r5 = Counter(Y[k] for k, t in enumerate(T)
                 if any(low_alpha(l) and not P.UU_DATA.match(l) and re.search(r'[a-z]{3,}', l, re.I)
                        for l in body_of(t).split('\n')))
print(f'    含单词的低字母占比行（按字母占比删乱码会误删）：{sum(old_r5.values())} 个帖子，其中棒球 {old_r5[5]} 个')

inner = sum(1 for t in T if P.strip_inner_header(body_of(t).split('\n')) != body_of(t).split('\n'))
print(f'    正文开头的内层头部：{inner} 个帖子')

eaten = sum(1 for t in T
            if any(P.UU_DATA.match(l) and any(P.SIG_SEP.match(x) for x in ls[k + 1:k + 4])
                   for ls in [body_of(t).split('\n')] for k, l in enumerate(ls)))
print(f'    编码数据行之后 3 行内出现 "--"（尾行不做长度校验会误删）：{eaten} 个帖子')

# ---------------------------------------------------------------- [4]
print('\n[4] 数字的类别区分度（每类中出现数字的帖子数明显高于其他类）')
cnt = {c: Counter() for c in range(10)}
for k, t in enumerate(body):
    for tok in set(re.findall(r'\b\d{2,5}\b', t)):
        cnt[Y[k]][tok] += 1
per_class = Counter(Y)
for c in range(10):
    top = [f'{tok}({n})' for tok, n in cnt[c].most_common(40)
           if n >= 20 and n / per_class[c] > 3 * (sum(cnt[o][tok] for o in range(10) if o != c)
                                                   / (len(T) - per_class[c]) + 1e-9)]
    if top:
        print(f'    {P.CLASS_NAMES[c]:6s}: {" ".join(top[:6])}')

# ---------------------------------------------------------------- [5]
print('\n[5] 各类前 10 高频词中是否含头部/邮箱噪声（edu、com、writes、article、organization 等）')
noise = {'edu', 'com', 'writes', 'article', 'organization', 'lines', 'subject', 'university'}
for name, docs in [('清洗前', T), ('清洗后', body)]:
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words='english')
    M = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    tops = [terms[np.asarray(M[Y == c].mean(0)).ravel().argsort()[::-1][:10]] for c in range(10)]
    print(f'    {name}：{sum(bool(set(t) & noise) for t in tops)}/10 个类别含噪声词；词表 {len(terms):,} 个词')
