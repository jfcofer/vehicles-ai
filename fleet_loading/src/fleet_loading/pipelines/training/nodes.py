from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

NUMERIC_FEATURES = [
    "cu",
    "iso_week_sin",
    "iso_week_cos",
    "n_vehicles_in_episode",
    "n_trucks_in_episode",
    "total_cu_in_episode",
    "cu_to_capacity_ratio",
    "excess_cu",
    "max_cu_in_episode",
    "count_large_vehicles",
    "episode_needs_deferral",
    "cu_desc_rank",
    "fits_without_me",
    "candidate_rank",
]
CATEGORICAL_FEATURES = ["canton", "clase"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "loaded"


def _compute_defer_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, pos_label=False, zero_division=0)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])


def _greedy_pack_fits(
    cus: list[float], capacities: list[float]
) -> bool:
    """Check if all vehicles fit into trucks via first-fit decreasing."""
    remaining = [c for c in capacities]
    for cu in sorted(cus, reverse=True):
        placed = False
        for i in range(len(remaining)):
            if cu <= remaining[i]:
                remaining[i] -= cu
                placed = True
                break
        if not placed:
            return False
    return True


def encode_features(
    vehicles: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    df = vehicles.merge(
        episodes[["episode_id", "iso_week", "n_trucks", "truck_capacities"]],
        on="episode_id",
        how="left",
    )

    df["iso_week"] = df["iso_week"].astype(float)
    df["iso_week_sin"] = np.sin(2 * np.pi * df["iso_week"] / 52)
    df["iso_week_cos"] = np.cos(2 * np.pi * df["iso_week"] / 52)

    ep_sizes = df.groupby("episode_id")["cu"].transform("size")
    df["n_vehicles_in_episode"] = ep_sizes

    df["n_trucks_in_episode"] = df["n_trucks"].astype(float)
    df["total_cu_in_episode"] = df.groupby("episode_id")["cu"].transform("sum")

    total_capacity = df["n_trucks_in_episode"] * 6.0
    df["cu_to_capacity_ratio"] = df["total_cu_in_episode"] / total_capacity
    df["excess_cu"] = (df["total_cu_in_episode"] - total_capacity).clip(lower=0)

    df["max_cu_in_episode"] = df.groupby("episode_id")["cu"].transform("max")
    df["count_large_vehicles"] = df.groupby("episode_id")["cu"].transform(
        lambda x: (x >= 4.0).sum()
    )

    df["episode_needs_deferral"] = (
        df["total_cu_in_episode"] > total_capacity
    ).astype(float)

    df["cu_desc_rank"] = df.groupby("episode_id")["cu"].rank(
        ascending=False, method="first"
    )

    def compute_fits_without_me(group: pd.DataFrame) -> pd.Series:
        capacities = group.iloc[0]["truck_capacities"]
        cus = group["cu"].tolist()
        result = []
        for i, cu in enumerate(cus):
            others = cus[:i] + cus[i+1:]
            result.append(_greedy_pack_fits(others, capacities))
        return pd.Series(result, index=group.index)

    df["fits_without_me"] = (
        df.groupby("episode_id", group_keys=False)
        .apply(compute_fits_without_me, include_groups=False)
        .astype(float)
    )

    def compute_candidate_rank(group: pd.DataFrame) -> pd.Series:
        fits = group["fits_without_me"].values
        cu_vals = group["cu"].values
        result = np.zeros(len(group))
        if fits.sum() == 0:
            return pd.Series(result, index=group.index)
        sorted_idx = np.argsort(cu_vals[fits == 1])
        ranks = np.argsort(sorted_idx)
        result[fits == 1] = (ranks + 1) / len(ranks)
        return pd.Series(result, index=group.index)

    df["candidate_rank"] = (
        df.groupby("episode_id", group_keys=False)
        .apply(compute_candidate_rank, include_groups=False)
        .astype(float)
    )

    df["loaded"] = (df["truck"] != "SIN_CAMION").astype(int)

    return df.drop(columns=["truck_capacities"]).reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes = df[["episode_id"]].drop_duplicates()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )
    train_idx, val_idx = next(
        splitter.split(episodes, groups=episodes["episode_id"])
    )

    train_ep = episodes.iloc[train_idx]["episode_id"]
    val_ep = episodes.iloc[val_idx]["episode_id"]

    train_df = df[df["episode_id"].isin(train_ep)].reset_index(drop=True)
    val_df = df[df["episode_id"].isin(val_ep)].reset_index(drop=True)

    return train_df, val_df


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: int,
    scale_pos_weight: float,
    max_delta_step: int,
    run_name: str,
) -> dict:
    import xgboost as xgb

    params = {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "scale_pos_weight": scale_pos_weight,
        "max_delta_step": max_delta_step,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    X_train_raw = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_val_raw = val_df[ALL_FEATURES]
    y_val = val_df[TARGET]

    preprocessor = _build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
        mlflow.log_param("xgb_preprocessor", "OrdinalEncoder(canton, clase) + passthrough(num)")

        y_pred = pipe.predict(X_val_raw)
        acc = accuracy_score(y_val, y_pred)
        f1 = _compute_defer_f1(y_val, y_pred)

        mlflow.log_metric("xgb_val_accuracy", acc)
        mlflow.log_metric("xgb_val_defer_f1", f1)
        mlflow.sklearn.log_model(
            pipe, "model",
            serialization_format="pickle",
        )

        return {
            "xgb_val_accuracy": acc,
            "xgb_val_defer_f1": f1,
        }


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_leaves: int,
    learning_rate: float,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
    min_child_samples: int,
    scale_pos_weight: float,
    run_name: str,
) -> dict:
    import lightgbm as lgb

    params = {
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_samples": min_child_samples,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary",
        "verbosity": -1,
    }

    X_train_raw = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_val_raw = val_df[ALL_FEATURES]
    y_val = val_df[TARGET]

    preprocessor = _build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)],
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({f"lgb_{k}": v for k, v in params.items()})
        mlflow.log_param("lgb_preprocessor", "OrdinalEncoder(canton, clase) + passthrough(num)")

        y_pred = pipe.predict(X_val_raw)
        acc = accuracy_score(y_val, y_pred)
        f1 = _compute_defer_f1(y_val, y_pred)

        mlflow.log_metric("lgb_val_accuracy", acc)
        mlflow.log_metric("lgb_val_defer_f1", f1)
        mlflow.sklearn.log_model(
            pipe, "model",
            serialization_format="pickle",
        )

        return {
            "lgb_val_accuracy": acc,
            "lgb_val_defer_f1": f1,
        }
