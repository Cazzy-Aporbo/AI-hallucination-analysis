#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Dungeon — surreal, game-style analysis for heavy-tailed media metrics.

What it does
------------
1) Sign-Flip Fate
   Bootstrap Spearman sign for each feature vs target (default: view_count).
   Output: probability of positive ("Aligned") vs negative ("Rebel") association.

2) Synergy Paradox Map
   For each pair, compute cross-validated R^2 for the duo and compare to a naive
   additive expectation from single-feature R^2 values (clipped at 1). Visualize
   ΔR^2 as a heatmap: rose = synergy, teal = anti-synergy.

3) Draft Royale
   Stochastic, budgeted drafts: each feature has a "uniqueness cost" based on
   redundancy (mean |ρ| with others). We form squads under a budget to maximize
   out-of-sample rank-R^2. Output: pick frequency across many drafts.

Usage
-----
python feature_dungeon.py --csv /path/to/data.csv --outdir ./artifacts --target view_count

Notes
-----
- Uses rank-based operations to be robust to heavy tails.
- Requires: numpy, pandas, scipy, scikit-learn, matplotlib
"""

import argparse
import os
import sys
import math
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


# ---------------- Palette (rose ↔ teal) ----------------
PALETTE = {
    "deep_plum": "#3f2f44",
    "rose": "#b37a86",
    "pale_rose": "#f4e9ec",
    "cool_teal": "#88b9c6",
    "ink": "#282633",
}

CMAP = LinearSegmentedColormap.from_list(
    "rose_teal_div",
    [PALETTE["cool_teal"], PALETTE["pale_rose"], PALETTE["rose"]],
    N=256,
)


def load_numeric_columns(csv_path: str, target_hint: str = "view_count") -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(csv_path)
    # Prefer these columns if present
    preferred = [
        "view_count",
        "duration_minutes",
        "virality_coefficient",
        "channel_follower_count",
        "title_length",
        "title_word_count",
    ]
    num_cols = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if target_hint not in num_cols:
        # try to infer a target
        vcand = [c for c in df.columns if target_hint in c and pd.api.types.is_numeric_dtype(df[c])]
        if vcand:
            # put the first found at front
            num_cols = [vcand[0]] + [c for c in num_cols if c != target_hint]

    # add more numeric columns (avoid obvious IDs)
    extra = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in num_cols and not any(k in c.lower() for k in ["id"])
    ]
    # keep it small-ish
    num_cols += extra[:8]
    # unique order
    num_cols = list(dict.fromkeys(num_cols))

    if not num_cols:
        raise ValueError("No numeric columns found in the CSV.")

    clean = df[num_cols].dropna().copy()

    # choose target
    target = target_hint if target_hint in clean.columns else clean.columns[0]
    features = [c for c in clean.columns if c != target]
    if len(features) < 3:
        raise ValueError(f"Need at least 4 numeric columns (target + >=3 features). Got: {len(clean.columns)}")

    return clean, target, features


def rankify(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(method="average")


def bootstrap_sign_prob(x: np.ndarray, y: np.ndarray, n_boot: int = 600, seed: int = 7) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    signs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        if np.isnan(r):
            r = 0.0
        signs.append(np.sign(r))
    signs = np.array(signs)
    p_pos = float((signs > 0).mean())
    p_neg = float((signs < 0).mean())
    return p_pos, p_neg


def cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 6, seed: int = 3) -> float:
    """Cross-validated R^2 using Pearson on predicted vs true ranks (squared)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = []
    model = LinearRegression()
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        yhat = model.predict(X[te])
        r = np.corrcoef(y[te], yhat)[0, 1]
        r2s.append((0 if np.isnan(r) else r) ** 2)
    return float(np.mean(r2s))


def sign_flip_fate(ranks: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
    y = ranks[target].to_numpy()
    stats = []
    for f in features:
        p_pos, p_neg = bootstrap_sign_prob(ranks[f].to_numpy(), y, n_boot=600, seed=7)
        stats.append((f, p_pos, p_neg))
    out = pd.DataFrame(stats, columns=["feature", "p_aligned", "p_rebel"]).set_index("feature")
    return out.sort_values("p_aligned", ascending=True)


def plot_sign_flip(sign_df: pd.DataFrame, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    y = np.arange(len(sign_df))
    ax.barh(y, sign_df["p_rebel"], color=PALETTE["cool_teal"], edgecolor=PALETTE["ink"], label="Rebel (negative)")
    ax.barh(y, sign_df["p_aligned"], left=sign_df["p_rebel"], color=PALETTE["rose"], edgecolor=PALETTE["ink"], label="Aligned (positive)")
    ax.set_yticks(y)
    ax.set_yticklabels(sign_df.index.tolist())
    ax.set_xlim(0, 1)
    ax.set_title("Sign-Flip Fate — Does this feature side with the View Boss?", color=PALETTE["deep_plum"], fontsize=13)
    ax.set_xlabel("Bootstrap probability")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)


def synergy_paradox(ranks: pd.DataFrame, target: str, features: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y = ranks[target].to_numpy().ravel()
    r2_single = {}
    for f in features:
        r2_single[f] = cv_r2(ranks[[f]].to_numpy(), y)

    synergy = pd.DataFrame(0.0, index=features, columns=features)
    for i, a in enumerate(features):
        for j, b in enumerate(features):
            if j <= i:
                continue
            Xab = ranks[[a, b]].to_numpy()
            r2_pair = cv_r2(Xab, y)
            expected = min(1.0, r2_single[a] + r2_single[b])  # naive additive expectation
            gain = r2_pair - expected
            synergy.loc[a, b] = gain
            synergy.loc[b, a] = gain
    return synergy, r2_single


def plot_synergy(synergy: pd.DataFrame, out_png: str) -> None:
    feats = synergy.index.tolist()
    fig, ax = plt.subplots(figsize=(8.2, 7.2), dpi=140)
    im = ax.imshow(synergy.values, cmap=CMAP, vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_title("Synergy Paradox Map — Combo gain beyond additive expectation", color=PALETTE["deep_plum"], fontsize=13)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("ΔR² (cross-validated)", rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)


def draft_royale(ranks: pd.DataFrame, target: str, features: List[str], r2_single: Dict[str, float], budget: float = 1.8, trials: int = 400, seed: int = 99) -> pd.Series:
    """Budgeted stochastic draft to assemble high-R^2 squads; returns pick frequency."""
    y = ranks[target].to_numpy().ravel()
    rho = ranks[[target] + features].corr(method="spearman")
    red = rho.loc[features, features].abs().replace(1.0, np.nan).mean(skipna=True)
    cost = red / red.max()

    rng = np.random.default_rng(seed)

    def cv_r2_feats(feats: List[str]) -> float:
        if not feats:
            return 0.0
        X = ranks[feats].to_numpy()
        return cv_r2(X, y)

    def draft_once(pool: List[str]) -> List[str]:
        picks, spent = [], 0.0
        base_r2 = 0.0
        while pool:
            best_gain, best = -1e9, None
            for f in pool:
                squad = picks + [f]
                gain = cv_r2_feats(squad) - base_r2
                gain += rng.normal(0, 0.01)  # chaos for exploration
                val = gain / (0.15 + cost[f])
                if val > best_gain and spent + cost[f] <= budget:
                    best_gain, best = val, f
            if best is None:
                break
            picks.append(best)
            spent += cost[best]
            base_r2 = cv_r2_feats(picks)
            pool.remove(best)
        return picks

    freq = pd.Series(0.0, index=features)
    for _ in range(trials):
        picks = draft_once(features[:])
        for f in picks:
            freq[f] += 1
    return freq / trials


def plot_draft(freq: pd.Series, out_png: str) -> None:
    freq = freq.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=140)
    ax.barh(freq.index, freq.values, color=PALETTE["rose"], edgecolor=PALETTE["ink"])
    ax.set_xlabel("Pick frequency in Draft Royale")
    ax.set_title("Draft Royale — Which features get drafted into winning squads?", color=PALETTE["deep_plum"], fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Feature Dungeon — surreal game-style analysis")
    ap.add_argument("--csv", required=True, help="Path to the CSV data (numeric columns).")
    ap.add_argument("--outdir", default="./artifacts", help="Output directory for plots and tables.")
    ap.add_argument("--target", default="view_count", help="Target column name (default: view_count).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    clean, target, features = load_numeric_columns(args.csv, target_hint=args.target)
    ranks = rankify(clean)

    # 1) Sign-Flip Fate
    sign_df = sign_flip_fate(ranks, target, features)
    plot_sign_flip(sign_df, os.path.join(args.outdir, "sign_flip_fate.png"))

    # 2) Synergy Paradox Map
    synergy, r2_single = synergy_paradox(ranks, target, features)
    plot_synergy(synergy, os.path.join(args.outdir, "synergy_paradox_map.png"))

    # 3) Draft Royale
    freq = draft_royale(ranks, target, features, r2_single, budget=1.8, trials=400, seed=99)
    plot_draft(freq, os.path.join(args.outdir, "draft_royale_picks.png"))

    # Summary table
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictive = pd.Series({f: abs(spearmanr(ranks[target], ranks[f]).correlation) for f in features})
    rho = ranks[[target] + features].corr(method="spearman")
    red = rho.loc[features, features].abs().replace(1.0, np.nan).mean(skipna=True)
    cost = red / red.max()

    summary = pd.DataFrame({
        "Predictive(|ρ|)": predictive.reindex(features),
        "Cost(uniqueness↓)": cost.reindex(features),
        "DraftFreq": freq.reindex(features),
        "pAligned": sign_df.reindex(features)["p_aligned"],
        "pRebel": sign_df.reindex(features)["p_rebel"],
    }).sort_values("DraftFreq", ascending=False)

    summary.to_csv(os.path.join(args.outdir, "feature_dungeon_summary.csv"), index=True)

    print(f"[Feature Dungeon] target={target}")
    print(f"Saved to: {args.outdir}")
    for f in ["sign_flip_fate.png", "synergy_paradox_map.png", "draft_royale_picks.png", "feature_dungeon_summary.csv"]:
        print(" -", os.path.join(args.outdir, f))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
