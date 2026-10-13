# -*- coding: utf-8 -*-
"""
五步流水线：数字化转型对企业经营绩效的影响研究
步骤：1 数据加载与模拟 -> 2 数据清洗 -> 3 变量构建 -> 4 信效度检验 -> 5 回归分析
运行：python3 pipeline.py
输出：data/panel_data.csv, results/*.csv, results/summary.txt
说明：本脚本使用模拟数据（随机种子固定，结果可复现），用于方法流程演示；
      实际研究中可将 step1 替换为 CSMAR/Wind 等数据库导出的真实数据。
"""
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
N_FIRM, YEARS = 300, 5  # 300家制造业上市公司, 2019-2023


def step1_load():
    """步骤1：数据加载（此处为模拟数据生成）"""
    firm_ids = np.repeat([f"S{i:04d}" for i in range(1, N_FIRM + 1)], YEARS)
    year = np.tile(np.arange(2019, 2024), N_FIRM)
    firm_effect = RNG.normal(0, 1, N_FIRM).repeat(YEARS)
    time_trend = (year - 2021) * 0.08 + RNG.normal(0, 0.05, len(year))
    # 数字化转型指数（词频法对数）
    dt = 1.2 + 0.5 * firm_effect + time_trend + RNG.normal(0, 0.6, len(year))
    dt = np.clip(dt, 0, None)
    # 量表题项（数字化转型感知，5题项, 1-7 Likert, 由 dt 生成）
    items = {}
    for k in range(1, 6):
        load = 0.75 + 0.05 * k
        items[f"DT{k}"] = np.clip(np.round(3.5 + load * dt * 0.8 + RNG.normal(0, 0.9, len(year))), 1, 7)
    # 控制变量
    size = np.clip(22.0 + firm_effect * 0.8 + RNG.normal(0, 1.0, len(year)), 19, 27)
    lev = np.clip(0.42 + 0.05 * firm_effect + RNG.normal(0, 0.15, len(year)), 0.05, 0.9)
    age = np.clip(18 + (year - 2021) * 1 + RNG.normal(10, 5, len(year)), 3, 60)
    rd = np.clip(0.04 + 0.01 * dt * 0.3 + RNG.normal(0, 0.02, len(year)), 0, 0.3)
    # 企业绩效 ROA = f(DT, 控制变量)
    roa = (0.031 + 0.018 * dt + 0.012 * (size - 22) - 0.035 * (lev - 0.42)
           + 0.001 * (age - 18) + 0.05 * rd + RNG.normal(0, 0.02, len(year)))
    df = pd.DataFrame({"firm": firm_ids, "year": year, "DT_index": dt,
                       "size": size, "lev": lev, "age": age, "rd": rd, "ROA": roa, **items})
    df.to_csv("data/panel_data.csv", index=False, encoding="utf-8-sig")
    return df


def step2_clean(df):
    """步骤2：数据清洗（缩尾去极值）"""
    for col in ["DT_index", "ROA", "size", "lev", "rd"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)
    return df.dropna().reset_index(drop=True)


def step3_vars(df):
    """步骤3：变量构建（量表均值、中心化）"""
    df["DT_scale"] = df[[f"DT{k}" for k in range(1, 6)]].mean(axis=1)
    df["DT_c"] = df["DT_index"] - df["DT_index"].mean()
    df["size_c"] = df["size"] - df["size"].mean()
    return df


def cronbach_alpha(d):
    k = d.shape[1]
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))


def step4_reliability(df):
    """步骤4：信效度检验（Cronbach's α + 题项总相关 CITC）"""
    items = df[[f"DT{k}" for k in range(1, 6)]]
    alpha = cronbach_alpha(items)
    rows = []
    for col in items.columns:
        rest = items.drop(columns=[col])
        rows.append([col, round(items[col].mean(), 3), round(items[col].std(), 3),
                     round(stats.pearsonr(items[col], rest.mean(axis=1))[0], 3),
                     round(cronbach_alpha(rest), 3)])
    pd.DataFrame(rows, columns=["题项", "均值", "标准差", "CITC", "删除后α"]).to_csv(
        "results/reliability.csv", index=False, encoding="utf-8-sig")
    # KMO
    corr = np.corrcoef(items.T)
    inv = np.linalg.pinv(corr)
    pcorr = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    np.fill_diagonal(pcorr, 1)
    kmo = (corr ** 2).sum() / ((corr ** 2).sum() + (pcorr ** 2).sum())
    with open("results/reliability_summary.txt", "w") as f:
        f.write(f"Cronbach's alpha = {alpha:.3f}\nKMO = {kmo:.3f}\n")
    return alpha, kmo


def step5_regression(df):
    """步骤5：回归分析（OLS，含固定效应的多元回归）"""
    X = df[["DT_c", "size_c", "lev", "age", "rd"]]
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"), X], axis=1)
    y = df["ROA"]
    Xv = X.values
    yv = y.values
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    dof = len(yv) - X.shape[1]
    se = np.sqrt(np.diag(XtX_inv) * (resid @ resid) / dof)
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    r2 = 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))
    adj_r2 = 1 - (1 - r2) * (len(yv) - 1) / dof
    out = pd.DataFrame({"变量": X.columns, "系数": beta.round(4), "标准误": se.round(4),
                        "t值": tval.round(3), "p值": pval.round(4)})
    out["显著性"] = out["p值"].apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "")))
    out.to_csv("results/regression.csv", index=False, encoding="utf-8-sig")
    # 分年度描述统计
    desc = df[["DT_index", "ROA", "size", "lev", "age", "rd"]].describe().round(4)
    desc.to_csv("results/descriptive.csv", encoding="utf-8-sig")
    corr_m = df[["DT_index", "ROA", "size", "lev", "age", "rd"]].corr().round(3)
    corr_m.to_csv("results/correlation.csv", encoding="utf-8-sig")
    with open("results/summary.txt", "a") as f:
        f.write(f"R2 = {r2:.4f}\nAdj R2 = {adj_r2:.4f}\nN = {len(y)}\n")
    return out, r2, adj_r2


if __name__ == "__main__":
    df = step1_load()
    df = step2_clean(df)
    df = step3_vars(df)
    alpha, kmo = step4_reliability(df)
    out, r2, adj_r2 = step5_regression(df)
    print(out.to_string(index=False))
    print(f"\nCronbach's alpha={alpha:.3f}, KMO={kmo:.3f}, R2={r2:.4f}, AdjR2={adj_r2:.4f}")
