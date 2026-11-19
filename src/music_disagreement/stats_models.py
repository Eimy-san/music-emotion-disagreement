from __future__ import annotations

from typing import Iterable, List

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def group_summary(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)[value_col]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


def correlation_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    return df[cols].corr(numeric_only=True)


def run_ols(df: pd.DataFrame, formula: str):
    return smf.ols(formula=formula, data=df).fit()


def run_robust_ols(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    design = df[[y_col] + x_cols].dropna()
    y = design[y_col]
    X = sm.add_constant(design[x_cols], has_constant="add")
    return sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()


def try_mixed_effects(df: pd.DataFrame, formula: str, group_col: str):
    if group_col not in df.columns:
        raise ValueError(f"Missing mixed-effects grouping column: {group_col}")
    return smf.mixedlm(formula=formula, data=df, groups=df[group_col]).fit()

