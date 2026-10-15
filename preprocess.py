"""实验一 第一部分：数据预处理

包含数据加载、验证集划分（约 75/25，分层且按讨论串分组）、文本清洗和清洗检查报告。
直接运行会生成划分文件和检查报告：

    python preprocess.py

第二部分通过 `from preprocess import prepare` 获取划分并清洗好的数据。

清洗只用逐条文本的固定规则，不从数据中学习任何统计量。
去停用词、去标点、TF-IDF 属于特征提取，放在第二部分，并且只在训练部分上 fit。
各项决策的缘由见 ../数据预处理决策.md。
"""
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path(__file__).resolve().parent
SPLIT_DIR = DATA_DIR / 'splits'
OUTPUT_DIR = DATA_DIR / 'outputs'
SEED = 42
VAL_SIZE = 0.25
# 根据各类高频词推断的类别含义（非官方），仅用于让报告更易读
CLASS_NAMES = ['图形', 'Windows', 'PC硬件', '汽车', '摩托车', '棒球', '加密', '电子', '医学', '中东政治']


# ---------------------------------------------------------------- 加载与划分

def load_data():
    """读取训练集和无标签测试集。沿用 draft_main.py，路径改为相对本文件，在任何目录下运行都能找到"""
    train_df = pd.read_csv(DATA_DIR / 'train_data.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def thread_ids(texts):
    """讨论串编号：标题去掉 Re:/Fwd: 前缀、合并空白、转小写后相同的帖子属于同一讨论串。
    只用于决定划分时哪些帖子必须放在同一边，不作为模型特征（模型只用正文）"""
    ids = []
    for t in texts:
        m = SUBJECT.search(split_header(t)[0])
        ids.append(re.sub(r'\s+', ' ', RE_PREFIX.sub('', m.group(1) if m else '')).strip().lower())
    return np.array(ids, dtype=object)


def split_path(val_size=VAL_SIZE):
    return SPLIT_DIR / f'split_val{round(val_size * 100)}_group.json'


def get_split(X, y, val_size=VAL_SIZE):
    """按讨论串分组、同时分层的训练/验证划分：同一讨论串的帖子只会整体落在一边。
    同一讨论串的帖子话题相同、参与者重叠，随机划分会让信息从训练部分漏到验证集（PDF 第 3 页：存在群组依赖时应分组划分）。
    用不打乱的 StratifiedGroupKFold 取第 0 折：打乱会破坏它的分层算法（实测类别占比最大偏离 25% 达 8～13 个点，
    不打乱时 ≤ 0.1 个点），因此划分是确定的，不依赖随机种子。
    首次调用时生成并保存，之后直接读取，保证所有实验共用同一份验证集"""
    path = split_path(val_size)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['n_samples'] != len(y):
            raise ValueError(f'{path.name} 对应 {saved["n_samples"]} 条样本，当前数据有 {len(y)} 条')
        return np.array(saved['train_idx']), np.array(saved['val_idx'])
    groups = thread_ids(X)
    splitter = StratifiedGroupKFold(n_splits=round(1 / val_size), shuffle=False)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y, groups))   # 取第 0 折作为验证集
    train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
    SPLIT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        'val_size': val_size, 'n_samples': len(y), 'grouped_by': 'thread (normalized Subject)',
        'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(),
    }), encoding='utf-8')
    return train_idx, val_idx


# ---------------------------------------------------------------- 清洗规则

@dataclass(frozen=True)
class CleanConfig:
    """每一步都能单独开关，用于消融实验。默认值就是最终方案"""
    header: bool = True          # ① 删除头部，以及正文开头的内层头部
    keep_subject: bool = False   # ① 的例外：保留标题。已决定不用，只为复现标题实验而保留
    quotes: bool = True          # ② 删除 > 和 |> 开头的引用行
    attribution: bool = True     # ③ 删除 "xxx writes:" 这类引述行
    uuencode: bool = True        # ⑤ 删除 uuencode 编码的二进制内容
    signature: bool = True       # ④ 删除标准 "--" 分隔符之后的签名档
    email_url: bool = True       # ⑥ 删除邮箱和网址


DEFAULT_CFG = CleanConfig()
RAW_CFG = CleanConfig(header=False, quotes=False, attribution=False,
                      uuencode=False, signature=False, email_url=False)

# 头部和正文之间的分隔行：允许含空格或制表符（row 1740 的分隔行里有一个空格）
HEADER_SEP = re.compile(r'\n[ \t]*\n')
HEADER_FIELD = re.compile(r'^[ \t]*(From|Subject|Organization|Lines|Reply-To|NNTP-Posting-Host|Distribution|'
                          r'Article-I\.D\.|Date|Newsgroups|Message-ID|Sender|Keywords|Summary|Expires|'
                          r'References|X-[\w-]+)[ \t]*:', re.I)
SUBJECT = re.compile(r'^Subject:[ \t]*(.*)$', re.M | re.I)
RE_PREFIX = re.compile(r'^(\s*(re|fwd?)\s*:\s*)+', re.I)
# 单独的 | 不算引用：审计发现它常用于表格和矩阵（如 row 3911）
QUOTE = re.compile(r'^[ \t]*(>|\|[ \t]*>)')
ATTRIBUTION = re.compile(r'^.*(writes|wrote|says)[ \t]*:[ \t]*$', re.I)
# 只认标准签名分隔符 "--"，不认 "-----" 这类普通分隔线（row 920 的正文就在分隔线之后）
SIG_SEP = re.compile(r'^--[ \t]*$')
SIG_MAX_LINES = 15
# uuencode：begin 行、以 M 开头的 61 字符数据行（不含小写字母），以及块末尾的短数据行、` 行和 end 行
UU_BEGIN = re.compile(r'^begin [0-7]{3,4} \S')
UU_DATA = re.compile(r'^M[\x20-\x60]{59,61}[ \t]*$')
UU_CHARS = re.compile(r'[\x20-\x60]+')
UU_TAIL_WINDOW = 3
EMAIL_URL = re.compile(r'\S+@\S+|https?://\S+|www\.\S+')


def _record(log, rule, removed):
    """记录被删除的内容，供检查报告使用。log 为 None 时不记录"""
    if log is not None and removed:
        log[rule].append(removed)


def split_header(text):
    """在第一个只含空白的行处切开，返回 (头部, 正文)。找不到分隔行时视为没有头部，不删任何内容"""
    m = HEADER_SEP.search(text)
    return ('', text) if m is None else (text[:m.start()], text[m.end():])


def strip_inner_header(lines, log=None):
    """正文开头可能还有第二层头部（如转贴的帖子），连同夹在中间的空行一起删除"""
    k = 0
    while k < len(lines) and (not lines[k].strip() or HEADER_FIELD.match(lines[k])):
        k += 1
    if not any(HEADER_FIELD.match(line) for line in lines[:k]):
        return lines
    _record(log, 'inner_header', '\n'.join(lines[:k]).strip())
    return lines[k:]


def drop_lines(lines, pattern, rule, log=None):
    """删除匹配 pattern 的行"""
    kept = []
    for line in lines:
        if pattern.match(line):
            _record(log, rule, line)
        else:
            kept.append(line)
    return kept


def is_uu_tail(line):
    """编码块末尾的短数据行：首字符表示本行字节数 n，行长必须等于 1 + 4*ceil(n/3)（部分编码器多 1 个校验字符）。
    靠长度校验，才不会把紧跟在编码块后面的 "-- " 签名分隔符或全大写短句当成乱码（row 3642）"""
    if line.strip() and set(line.strip()) == {'`'} and len(line.strip()) <= 2:
        return True
    line = line.rstrip('\r')
    if not line or line[0] in ' `' or not UU_CHARS.fullmatch(line):
        return False
    expected = 1 + 4 * ((ord(line[0]) - 32 + 2) // 3)
    return len(line) in (expected, expected + 1)


def strip_uuencode(lines, log=None):
    """删除 uuencode 编码块。
    begin 行和标准数据行直接删；块末尾的短数据行、` 行和 end 行，只在紧跟数据行之后的几行内才删，
    避免把正文里恰好只有一个 "end" 的行误删"""
    kept, tail = [], 0
    for line in lines:
        s = line.rstrip()
        if UU_BEGIN.match(s) or UU_DATA.match(line):
            _record(log, 'uuencode', line)
            tail = UU_TAIL_WINDOW
            continue
        if tail > 0:
            tail -= 1
            if s == 'end' or is_uu_tail(line):
                _record(log, 'uuencode', line)
                if s == 'end':
                    tail = 0
                continue
        kept.append(line)
    return kept


def strip_signature(lines, log=None):
    """从最后一个标准 "--" 分隔行处截断。
    只在其后不超过 SIG_MAX_LINES 行、并且截断后正文不为空时才截，防止把正文当签名删掉"""
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    for i in range(len(lines) - 1, -1, -1):
        if SIG_SEP.match(lines[i]):
            if len(lines) - i <= SIG_MAX_LINES and any(line.strip() for line in lines[:i]):
                _record(log, 'signature', '\n'.join(lines[i:]))
                return lines[:i]
            break
    return lines


def clean_text(text, cfg=DEFAULT_CFG, log=None):
    """按配置依次执行清洗规则。log 传入一个 defaultdict(list) 时，会记录每条规则删掉的内容"""
    if cfg.header:
        header, body = split_header(text)
        _record(log, 'header', header)
        lines = strip_inner_header(body.split('\n'), log)
    else:
        header, lines = '', text.split('\n')
    if cfg.quotes:
        lines = drop_lines(lines, QUOTE, 'quotes', log)
    if cfg.attribution:
        lines = drop_lines(lines, ATTRIBUTION, 'attribution', log)
    if cfg.uuencode:
        lines = strip_uuencode(lines, log)
    if cfg.signature:
        lines = strip_signature(lines, log)
    body = '\n'.join(lines)
    if cfg.header and cfg.keep_subject:
        m = SUBJECT.search(header)
        if m:
            body = RE_PREFIX.sub('', m.group(1)).strip() + '\n' + body
    if cfg.email_url:
        for match in EMAIL_URL.findall(body):
            _record(log, 'email_url', match)
        body = EMAIL_URL.sub(' ', body)
    return body


def prepare(cfg=DEFAULT_CFG, val_size=VAL_SIZE):
    """加载 → 划分 → 分别清洗训练、验证、测试。第二部分从这里取数据。
    按 PDF 规定的顺序先划分再清洗；清洗规则不从数据里学东西，三部分用的是同一套规则"""
    X, y, X_test = load_data()
    train_idx, val_idx = get_split(X, y, val_size)
    groups = thread_ids(X)
    clean = lambda docs: [clean_text(t, cfg) for t in docs]
    return {
        'X_train': clean([X[i] for i in train_idx]), 'y_train': y[train_idx],
        'X_val': clean([X[i] for i in val_idx]), 'y_val': y[val_idx],
        'X_test': clean(X_test),
        'train_idx': train_idx, 'val_idx': val_idx,
        'groups_train': groups[train_idx], 'groups_val': groups[val_idx],   # 讨论串编号，用于分组交叉验证和分组 bootstrap
    }


# ---------------------------------------------------------------- 检查报告

RULES = ['header', 'inner_header', 'quotes', 'attribution', 'uuencode', 'signature', 'email_url']
RULE_NAMES = {'header': '① 头部', 'inner_header': '① 内层头部', 'quotes': '② 引用行',
              'attribution': '③ 引述行', 'uuencode': '⑤ uuencode', 'signature': '④ 签名档',
              'email_url': '⑥ 邮箱/网址'}


def _clean_with_log(texts, cfg):
    cleaned, logs = [], []
    for t in texts:
        log = defaultdict(list)
        cleaned.append(clean_text(t, cfg, log))
        logs.append(log)
    return cleaned, logs


def _short(s, n=110):
    s = s.replace('\n', ' ⏎ ')
    return repr(s[:n] + ('…' if len(s) > n else ''))


def cleaning_report(texts, y, train_idx, val_idx, test_texts, cfg=DEFAULT_CFG, seed=SEED):
    """返回 (摘要, 抽样明细) 两段文本。
    所有带标签的统计和被删内容抽样只用训练部分；验证集和测试集只核对格式是否一致，不看标签，不用于任何决策"""
    rng = random.Random(seed)
    rows = list(train_idx)                               # 训练部分在原始数据中的行号
    tr_texts, tr_y = [texts[i] for i in rows], y[train_idx]
    cleaned, logs = _clean_with_log(tr_texts, cfg)
    assert len(cleaned) == len(tr_texts), '清洗前后样本数不一致'
    raw_chars = sum(len(t) for t in tr_texts)
    summary, details = [], []
    out = summary.append

    out('=' * 72 + '\n实验一 数据预处理检查报告\n' + '=' * 72)
    out('\n[1] 清洗配置\n    ' + ', '.join(f'{k}={v}' for k, v in asdict(cfg).items()))

    out('\n[2] 验证集划分（在设计清洗规则之前完成）')
    no_overlap = len(set(train_idx) & set(val_idx)) == 0
    full_cover = len(set(train_idx) | set(val_idx)) == len(y)
    out(f'    全部 {len(y)} 条 → 训练 {len(train_idx)} 条 / 验证 {len(val_idx)} 条（验证占 {len(val_idx) / len(y):.1%}），'
        f'两部分无重叠：{no_overlap}，合起来覆盖全部样本：{full_cover}')
    groups = thread_ids(texts)
    g_tr, g_va = set(groups[train_idx]), set(groups[val_idx])
    out(f'    按讨论串分组：训练 {len(g_tr)} 个讨论串 / 验证 {len(g_va)} 个，同时出现在两边的讨论串：{len(g_tr & g_va)} 个')
    out(f'    {"类别":<10s}{"全部":>6s}{"训练":>6s}{"验证":>6s}{"验证占比":>9s}')
    c_all, c_tr, c_va = Counter(y), Counter(y[train_idx]), Counter(y[val_idx])
    for c in sorted(c_all):
        out(f'    {c} {CLASS_NAMES[c]:<8s}{c_all[c]:>6d}{c_tr[c]:>6d}{c_va[c]:>6d}{c_va[c] / c_all[c]:>9.1%}')

    out(f'\n[3] 各规则删除量（只用训练部分 {len(tr_texts)} 条）')
    out(f'    清洗前后样本数：{len(tr_texts)} → {len(cleaned)}，没有删除样本，标签未改动')
    out(f'    {"规则":<12s}{"涉及帖子":>14s}{"删除条目":>10s}{"删除字符":>12s}{"占原文字符":>10s}')
    for r in RULES:
        posts = sum(1 for lg in logs if lg[r])
        items = sum(len(lg[r]) for lg in logs)
        chars = sum(len(s) for lg in logs for s in lg[r])
        out(f'    {RULE_NAMES[r]:<12s}{posts:>6d} ({posts / len(tr_texts):5.1%}){items:>10d}'
            f'{chars:>12,d}{chars / raw_chars:>10.1%}')
    out('    各类别中受影响的帖子数：')
    out('    ' + ' ' * 12 + ''.join(f'{CLASS_NAMES[c]:>8s}' for c in range(10)))
    for r in RULES:
        per = Counter(tr_y[k] for k, lg in enumerate(logs) if lg[r])
        out(f'    {RULE_NAMES[r]:<12s}' + ''.join(f'{per[c]:>8d}' for c in range(10)))

    out('\n[4] 文本长度变化（训练部分，字符数）')
    before = pd.Series([len(t) for t in tr_texts])
    after = pd.Series([len(t) for t in cleaned])
    out(f'    清洗前：总计 {before.sum():,}，中位数 {int(before.median())}，均值 {int(before.mean())}')
    out(f'    清洗后：总计 {after.sum():,}，中位数 {int(after.median())}，均值 {int(after.mean())}'
        f'（保留 {after.sum() / before.sum():.1%}）')

    out('\n[5] 清洗后为空的样本（训练部分；保留，不删除）')
    empty = [k for k, t in enumerate(cleaned) if not t.strip()]
    per = Counter(tr_y[k] for k in empty)
    out(f'    {len(empty)} 条（{len(empty) / len(tr_texts):.2%}），行号 {[int(rows[k]) for k in empty]}')
    out('    按类别：' + '，'.join(f'{CLASS_NAMES[c]} {per[c]}' for c in range(10) if per[c]))

    out('\n[6] 验证集、测试集的格式一致性核对（不看标签，不用于任何决策）')
    for name, part in [('验证集', [texts[i] for i in val_idx]), ('测试集', test_texts)]:
        part_cleaned, part_logs = _clean_with_log(part, cfg)
        n_empty = sum(1 for t in part_cleaned if not t.strip())
        cover = '，'.join(f'{RULE_NAMES[r]} {sum(1 for lg in part_logs if lg[r]) / len(part):.1%}' for r in RULES)
        out(f'    {name} {len(part)} 条：清洗后为空 {n_empty} 条；各规则涉及的帖子比例：{cover}')
    tr_cover = '，'.join(f'{RULE_NAMES[r]} {sum(1 for lg in logs if lg[r]) / len(tr_texts):.1%}' for r in RULES)
    out(f'    （对照）训练部分：{tr_cover}')

    add = details.append
    add('\n[7] 被删内容抽样（训练部分）：每条规则在每个类别随机抽 1 条，用于人工核对是否误删')
    for r in RULES:
        add(f'\n  --- {RULE_NAMES[r]} ---')
        for c in range(10):
            pool = [(k, s) for k, lg in enumerate(logs) if tr_y[k] == c for s in lg[r]]
            if pool:
                k, s = rng.choice(pool)
                add(f'    [{CLASS_NAMES[c]}] row {rows[k]:>4d}: {_short(s)}')

    add('\n[8] 清洗前后对照（训练部分）：随机 3 条')
    for k in rng.sample(range(len(tr_texts)), 3):
        add(f'\n  --- row {rows[k]}，类别 {tr_y[k]} {CLASS_NAMES[tr_y[k]]} ---')
        add(f'    清洗前：{_short(tr_texts[k], 400)}')
        add(f'    清洗后：{_short(cleaned[k], 400)}')

    return '\n'.join(summary), '\n'.join(details)


def main():
    X, y, X_test = load_data()
    train_idx, val_idx = get_split(X, y)
    summary, details = cleaning_report(X, y, train_idx, val_idx, X_test)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / 'cleaning_report.txt'
    path.write_text(summary + '\n' + details + '\n', encoding='utf-8')
    print(summary)
    print(f'\n完整报告（含被删内容抽样）：{path}')


if __name__ == '__main__':
    main()
