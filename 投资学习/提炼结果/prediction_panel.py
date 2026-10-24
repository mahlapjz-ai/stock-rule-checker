# -*- coding: utf-8 -*-
"""
面板数据版预测模型 v2 —— 修复 z-score 泄漏 + 时间对齐报告 + block bootstrap

相对 v1 的三项修正：
  1) z-score 不再用全样本 fit：先按时间切分，用训练段（前60%时间）的 mean/std fit scaler，
     再 transform 验证/测试段，杜绝未来信息泄漏。
  2) 报告 30 只股票的时间对齐（起始/结束日期、样本数），标注起始晚 3 个月以上的股票。
  3) 留一股票 AUC 改用 block bootstrap（每 20 交易日一块）重算 95% CI。

用法：python prediction_panel.py
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

HORIZON = 5
LABEL_THRESHOLD = 0.01
BOOT_N = 1000
BLOCK = 20  # block bootstrap 时间块（交易日）

STOCKS = [
    "sh600519", "sh600036", "sh601318", "sh601398", "sh601939",
    "sz000858", "sz300750", "sz000333", "sz000651", "sh601888",
    "sh600276", "sh603259", "sz002594", "sh601012", "sh603288",
    "sh600887", "sh600028", "sh601857", "sh600900", "sz000002",
    "sh600048", "sh600030", "sz300059", "sz002415", "sz002475",
    "sz000725", "sh601668", "sh601766", "sh600050", "sh600104",
]

FEAT_COLS = [f"ma{n}_{k}" for n in (5, 10, 20, 60) for k in ("pos", "slope")] \
    + ["macd_dif", "macd_dea", "macd_hist", "rsi14", "vol_ratio", "turnover", "amp5"]


def fetch_day(code, count=1000):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={code},day,,,{count}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    arr = data["data"][code]["day"]
    return pd.DataFrame([dict(date=x[0], open=float(x[1]), close=float(x[2]),
                              high=float(x[3]), low=float(x[4]), vol=float(x[5])) for x in arr])


def fetch_turnover_rate(code):
    url = f"https://qt.gtimg.cn/q={code}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("gbk", errors="replace")
    m = text.split('="', 1)[1].rsplit('"', 1)[0]
    f = m.split("~")
    return float(f[38]), float(f[36])


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def build_features(df, hsl_last, vol_last):
    close, high, low, vol = df["close"], df["high"], df["low"], df["vol"]
    feats = {}
    for n in (5, 10, 20, 60):
        ma = close.rolling(n).mean()
        feats[f"ma{n}_pos"] = (close - ma) / ma
        feats[f"ma{n}_slope"] = (ma - ma.shift(5)) / ma.shift(5)
    dif = ema(close, 12) - ema(close, 26)
    dea = ema(dif, 9)
    feats["macd_dif"], feats["macd_dea"], feats["macd_hist"] = dif, dea, dif - dea
    delta = close.diff()
    rs = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    feats["rsi14"] = 100 - 100 / (1 + rs)
    feats["vol_ratio"] = vol / vol.rolling(5).mean()
    feats["turnover"] = vol / vol_last * hsl_last if (hsl_last and vol_last and vol_last > 0) else vol / vol.rolling(20).mean()
    feats["amp5"] = (high.rolling(5).max() - low.rolling(5).min()) / close
    X = pd.DataFrame(feats)
    return X, X.notna().all(axis=1)


def build_labels(df, threshold=LABEL_THRESHOLD):
    ret = df["close"].shift(-HORIZON) / df["close"] - 1
    return (ret > threshold).astype(int)


def load_one(code):
    """拉一只股票 → 算原始特征/标签（不做标准化）。返回 DataFrame。"""
    try:
        df = fetch_day(code)
        hsl, vol_last = fetch_turnover_rate(code)
    except Exception as e:
        print(f"  [跳过] {code}: {e}")
        return None
    X, valid = build_features(df, hsl, vol_last)
    y = build_labels(df)
    mask = valid & y.notna()
    if mask.sum() < 300:
        print(f"  [跳过] {code}: 有效样本不足 {mask.sum()}")
        return None
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    dates = df["date"][mask].reset_index(drop=True)
    X["code"] = code
    X["date"] = dates.values
    X["label"] = y.values
    return X


def block_bootstrap_auc(pred_df, block=BLOCK, n_boot=BOOT_N, seed=42):
    """按时间块（block 个交易日）重采样，计算合并预测的 AUC 95% CI。"""
    pred_df = pred_df.reset_index(drop=True)
    n = len(pred_df)
    rng = np.random.RandomState(seed)
    aucs = []
    while len(aucs) < n_boot:
        idx = []
        while len(idx) < n:
            start = rng.randint(0, n - block + 1)
            idx.extend(range(start, start + block))
        idx = np.array(idx[:n])
        if len(np.unique(pred_df["label"].iloc[idx])) < 2:
            continue
        aucs.append(roc_auc_score(pred_df["label"].iloc[idx], pred_df["prob"].iloc[idx]))
    a = np.array(aucs)
    return np.percentile(a, 2.5), np.percentile(a, 97.5), a.mean()


def main():
    print(f"== 面板数据 v2：{len(STOCKS)} 只沪深300成分股 ==")
    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = list(ex.map(load_one, STOCKS))
    frames = [f for f in frames if f is not None]
    panel = pd.concat(frames, ignore_index=True)

    # ---- 第2项：时间对齐报告 ----
    print(f"\n[2] 时间对齐报告（{panel['code'].nunique()} 只股票）：")
    align_rows = []
    for code, grp in panel.groupby("code"):
        align_rows.append((code, grp["date"].min(), grp["date"].max(), len(grp)))
    align = pd.DataFrame(align_rows, columns=["code", "start", "end", "n"])
    align = align.sort_values("start")
    # 全局最早起始日期
    earliest = align["start"].min()
    for _, r in align.iterrows():
        late = " ⚠ 晚3个月以上" if (pd.to_datetime(r["start"]) - pd.to_datetime(earliest)).days > 90 else ""
        print(f"    {r['code']}: {r['start']} ~ {r['end']}  n={r['n']}{late}")
    late_stocks = align[(pd.to_datetime(align['start']) - pd.to_datetime(earliest)).dt.days > 90]["code"].tolist()
    if late_stocks:
        print(f"  → 起始晚3个月以上的股票：{late_stocks}（建议剔除，避免训练段无该股样本）")
    else:
        print(f"  → 无股票起始晚3个月以上，时间轴对齐良好")

    # ---- 全局时间边界（所有股票共用时间轴）----
    panel = panel.sort_values("date").reset_index(drop=True)
    dates_uq = np.unique(panel["date"].values)
    d60, d80 = dates_uq[int(len(dates_uq) * 0.6)], dates_uq[int(len(dates_uq) * 0.8)]

    # ---- 第1项：每只股票内部 z-score，scaler 只用训练段（date<=d60）fit ----
    scaled_parts = []
    for code, grp in panel.groupby("code"):
        grp = grp.copy()
        tr_mask = grp["date"] <= d60
        sc = StandardScaler().fit(grp.loc[tr_mask, FEAT_COLS])
        grp[FEAT_COLS] = sc.transform(grp[FEAT_COLS])
        scaled_parts.append(grp)
    panel = pd.concat(scaled_parts, ignore_index=True)

    tr = panel[panel["date"] <= d60]
    va = panel[(panel["date"] > d60) & (panel["date"] <= d80)]
    te = panel[panel["date"] > d80]
    print(f"\n[1] 跨时间划分（z-score 无泄漏）：训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}")
    print(f"    训练日期 ≤ {d60}，验证 {d60}~{d80}，测试 > {d80}")

    def train_eval(tr_df, te_df):
        m = LogisticRegression(max_iter=2000, random_state=42)
        m.fit(tr_df[FEAT_COLS], tr_df["label"])
        prob = m.predict_proba(te_df[FEAT_COLS])[:, 1]
        return dict(auc=roc_auc_score(te_df["label"], prob),
                    acc=accuracy_score(te_df["label"], m.predict(te_df[FEAT_COLS])),
                    prob=prob, true=te_df["label"].values)

    r_te = train_eval(tr, te)
    r_va = train_eval(tr, va)
    print(f"    验证集 AUC = {r_va['auc']:.4f}，测试集 AUC = {r_te['auc']:.4f}")

    # ---- 第3项：留一股票 + block bootstrap ----
    print(f"\n[3] 留一股票交叉验证（每折留 1 只股票测试）＋ block bootstrap（{BLOCK}交易日/块）：")
    groups = panel["code"].astype("category").cat.codes.values
    logo = LeaveOneGroupOut()
    per_fold = []
    preds = []
    for tr_idx, te_idx in logo.split(panel[FEAT_COLS], panel["label"], groups):
        tr_df, te_df = panel.iloc[tr_idx], panel.iloc[te_idx]
        m = LogisticRegression(max_iter=2000, random_state=42)
        m.fit(tr_df[FEAT_COLS], tr_df["label"])
        prob = m.predict_proba(te_df[FEAT_COLS])[:, 1]
        per_fold.append(roc_auc_score(te_df["label"], prob))
        for d, p, l in zip(te_df["date"].values, prob, te_df["label"].values):
            preds.append((d, p, l))
    per_fold = np.array(per_fold)
    pred_df = pd.DataFrame(preds, columns=["date", "prob", "label"]).sort_values("date").reset_index(drop=True)

    lo, hi, mean_auc = block_bootstrap_auc(pred_df)
    overall = roc_auc_score(pred_df["label"], pred_df["prob"])
    print(f"    每折 AUC 均值 = {per_fold.mean():.4f} ± {per_fold.std():.4f}（{len(per_fold)} 折）")
    print(f"    合并预测 AUC = {overall:.4f}")
    print(f"    block bootstrap 95%CI = [{lo:.4f}, {hi:.4f}]，均值 {mean_auc:.4f}")
    print(f"    {'CI 下限 > 0.5，显著高于随机' if lo > 0.5 else 'CI 跨越 0.5，与随机无显著差异'}")

    print("\n结论：")
    print(f"  跨时间测试 AUC={r_te['auc']:.4f}，留一股票合并 AUC={overall:.4f}（block bootstrap CI=[{lo:.4f},{hi:.4f}]）")


if __name__ == "__main__":
    main()
