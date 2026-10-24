# -*- coding: utf-8 -*-
"""
规则一致性检测器 —— 预测模型 v3

相对 v2 的三项改动：
  1) 除权日识别：改用「前复权/不复权价格比例突变」精确定位（不再用跳空>2%启发式）
  2) 模型对比：逻辑回归 + 随机森林 + XGBoost（若已安装）
  3) 标签：close[t+5]/close[t]-1 > 0.01（涨超 1% 才算正样本）

用法：python prediction_model.py [股票代码] [K线条数]
"""

import sys
import json
import urllib.request

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HORIZON = 5
EX_RATIO_CHG = 0.02   # 复权因子突变阈值（>2% 视为除权日）
EX_WINDOW = 5         # 除权日前后剔除窗口
BOOT_N = 1000
LABEL_THRESHOLD = 0.01  # 涨超 1% 才算正样本


# ----------------------------------------------------------------------
def fetch_day(code, count):
    """不复权日K。"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param={code},day,,,{count}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    arr = data["data"][code]["day"]
    return pd.DataFrame([dict(date=x[0], open=float(x[1]), close=float(x[2]),
                              high=float(x[3]), low=float(x[4]), vol=float(x[5])) for x in arr])


def fetch_qfq(code, count):
    """前复权日K（用于复权因子识别除权日）。"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{count},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    node = data["data"][code]
    arr = node.get("qfqday") or node.get("day")
    return pd.DataFrame([dict(date=x[0], close=float(x[2])) for x in arr])


def fetch_turnover_rate(code):
    url = f"https://qt.gtimg.cn/q={code}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("gbk", errors="replace")
    m = text.split('="', 1)[1].rsplit('"', 1)[0]
    f = m.split("~")
    return float(f[38]), float(f[36])


# ----------------------------------------------------------------------
# 除权日识别（复权因子突变法）
# ----------------------------------------------------------------------
def find_ex_dates(code, count):
    """前复权/不复权价格比例 ratio = day_close / qfq_close，除权日处 ratio 发生突变。"""
    day = fetch_day(code, count)
    qfq = fetch_qfq(code, count)
    m = day.merge(qfq, on="date", suffixes=("_d", "_q"))
    m["ratio"] = m["close_d"] / m["close_q"]
    m["chg"] = m["ratio"] / m["ratio"].shift(1)
    ex = m[abs(m["chg"] - 1) > EX_RATIO_CHG]
    return [(r["date"], round(r["chg"], 4)) for _, r in ex.iterrows()]


# ----------------------------------------------------------------------
# 特征 / 标签
# ----------------------------------------------------------------------
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
    return X, list(X.columns), X.notna().all(axis=1)


def build_labels(df, threshold=LABEL_THRESHOLD):
    """未来第5日收益率 > threshold 才算正样本。"""
    ret = df["close"].shift(-HORIZON) / df["close"] - 1
    return (ret > threshold).astype(int)


def remove_ex_window(df, ex_dates, window=EX_WINDOW):
    date_idx = {d: i for i, d in enumerate(df["date"])}
    drop = set()
    for d, _ in ex_dates:
        if d not in date_idx:
            continue
        c = date_idx[d]
        for k in range(-window, window + 1):
            j = c + k
            if 0 <= j < len(df):
                drop.add(j)
    keep = [i for i in range(len(df)) if i not in drop]
    return df.iloc[keep].reset_index(drop=True), len(drop)


def evaluate(y_true, y_pred, y_prob):
    return dict(auc=roc_auc_score(y_true, y_prob),
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                n=int(len(y_true)))


def bootstrap_auc_ci(y_true, y_prob, n_boot=BOOT_N, seed=42):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), aucs.mean()


# ----------------------------------------------------------------------
def time_split(X, y, dates):
    n = len(X)
    n1, n2 = int(n * 0.6), int(n * 0.8)
    return (X.iloc[:n1], y.iloc[:n1]), (X.iloc[n1:n2], y.iloc[n1:n2]), (X.iloc[n2:], y.iloc[n2:]), dates


def fit_model(name, tr, va, te):
    Xtr, ytr = tr; Xva, yva = va; Xte, yte = te
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)
    if name == "逻辑回归":
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif name == "随机森林":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    elif name == "XGBoost":
        model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric="logloss")
    model.fit(Xtr_s, ytr)
    def pp(X): return model.predict_proba(X)[:, 1]
    return dict(train=evaluate(ytr, model.predict(Xtr_s), pp(Xtr_s)),
                valid=evaluate(yva, model.predict(Xva_s), pp(Xva_s)),
                test=evaluate(yte, model.predict(Xte_s), pp(Xte_s)),
                test_prob=pp(Xte_s), test_true=yte)


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "sh600036"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"===== {code} · 预测模型 v3（复权因子除权 + 多模型 + 涨超1%标签）=====")
    df = fetch_day(code, count)
    hsl_last, vol_last = None, None
    try:
        hsl_last, vol_last = fetch_turnover_rate(code)
    except Exception as e:
        print(f"换手率快照获取失败({e})")

    # 第1项：复权因子法识别除权日
    ex = find_ex_dates(code, count)
    print(f"\n[1] 复权因子突变法识别除权日：共 {len(ex)} 个")
    for d, g in ex:
        print(f"    {d}  复权因子突变 {g:.4f}")

    # 特征+标签
    X, feat_names, valid = build_features(df, hsl_last, vol_last)
    y = build_labels(df)  # 涨超1%
    y_old = build_labels(df, threshold=0.0)  # 涨超0%（旧标签，用于对比）
    mask = valid & y.notna()
    X, y, y_old = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), y_old[mask].reset_index(drop=True)
    dates = df["date"][mask].reset_index(drop=True)
    n = len(X)
    print(f"\n有效样本 {n}，正样本(涨超1%)占比 {y.mean():.4f}，旧标签(涨超0%)占比 {y_old.mean():.4f}")

    # 除权剔除
    df_clean, dropped = remove_ex_window(df, ex)
    Xc, _, validc = build_features(df_clean, hsl_last, vol_last)
    yc = build_labels(df_clean)
    maskc = validc & yc.notna()
    Xc, yc = Xc[maskc].reset_index(drop=True), yc[maskc].reset_index(drop=True)

    # 划分
    tr, va, te, _ = time_split(X, y, dates)
    trc, vac, tec, _ = time_split(Xc, yc, dates)

    models = ["逻辑回归", "随机森林"] + (["XGBoost"] if HAS_XGB else [])
    print(f"\n[2] 模型对比（标签=涨超1%）：")
    for m in models:
        r = fit_model(m, tr, va, te)
        print(f"    {m}: 训练AUC={r['train']['auc']:.4f} 验证AUC={r['valid']['auc']:.4f} "
              f"测试AUC={r['test']['auc']:.4f} (n={r['test']['n']})")

    # 第3项：标签对比（逻辑回归，涨超0% vs 涨超1%）
    print(f"\n[3] 标签对比（逻辑回归）：")
    tr_old, va_old, te_old, _ = time_split(X, y_old, dates)
    r_old = fit_model("逻辑回归", tr_old, va_old, te_old)
    r_new = fit_model("逻辑回归", tr, va, te)
    print(f"    涨超0%标签：样本{n}，测试AUC={r_old['test']['auc']:.4f}")
    print(f"    涨超1%标签：样本{n}，测试AUC={r_new['test']['auc']:.4f}")

    # 除权剔除后的逻辑回归
    print(f"\n[1续] 剔除除权前后{EX_WINDOW}天重跑（逻辑回归，涨超1%标签）：")
    rc = fit_model("逻辑回归", trc, vac, tec)
    print(f"    原始样本 {n} → 剔除后 {len(Xc)}（减少 {dropped}），测试AUC {r_new['test']['auc']:.4f} → {rc['test']['auc']:.4f}")

    # 第3项 bootstrap CI（逻辑回归，涨超1%）
    lo, hi, mean_auc = bootstrap_auc_ci(r_new["test_true"], r_new["test_prob"])
    print(f"\n[bootstrap] 逻辑回归 测试AUC 95%CI = [{lo:.4f}, {hi:.4f}]，均值 {mean_auc:.4f}")

    print(f"\n结论：测试AUC(逻辑回归,涨超1%) = {r_new['test']['auc']:.4f}")
    if r_new['test']['auc'] < 0.52:
        print("   < 0.52，特征无预测力。")


if __name__ == "__main__":
    main()
