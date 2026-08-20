"""
합성 데이터(flight_price.db)로 Final_Model을 재학습하는 스크립트.

원본 Final_Model_notebook.ipynb와 동일한 피처 엔지니어링/모델 구조(RandomForest,
n_estimators=300, max_depth=12, 동일 feature_cols)를 쓰되, 소스를
(wide-format 엑셀이 아니라) 이미 long-format인 합성 flight_price.db로 바꿨습니다.

실제 회사 데이터는 전혀 쓰지 않으므로, 이 스크립트로 만든 모델과 이 저장소는
통째로 public이어도 안전합니다.

실행 방법:
    python build_synthetic_db.py     # 최초 1회: 합성 DB 생성
    python train_synthetic_model.py  # 합성 데이터로 모델 재학습 -> Final_Model.pkl 덮어씀
"""

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "flight_price.db"
MODEL_PATH = BASE_DIR / "Final_Model.pkl"

AIRLINES = ["KE", "OZ", "7C", "TW", "LJ", "TG"]
# 노트북과 동일한 값 (feature_importance가 0이라 실제 예측엔 영향 없음, 형식만 맞춤)
MARKET_SHARE = {"KE": 0.20, "OZ": 0.19, "7C": 0.16, "TW": 0.16, "LJ": 0.15, "TG": 0.14}


def flight_split(df, cut_ratio=0.8):
    train_parts, test_parts = [], []
    for _, g in df.groupby("flight_date"):
        g = g.sort_values("obs_date")
        cut = int(len(g) * cut_ratio)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def main():
    con = sqlite3.connect(DB_PATH)
    long_df = pd.read_sql(
        "SELECT flight_date, obs_date, days_before_departure, airline, price FROM flight_price", con
    )
    con.close()

    long_df["flight_date"] = pd.to_datetime(long_df["flight_date"])
    long_df["obs_date"] = pd.to_datetime(long_df["obs_date"])

    for a in AIRLINES:
        long_df[f"prev_share_{a}"] = MARKET_SHARE[a]

    wide_prev = long_df.pivot_table(
        index=["flight_date", "obs_date"], columns="airline", values="price"
    ).reset_index()
    wide_prev.columns.name = None
    wide_prev = wide_prev.rename(columns={a: f"prev_price_{a}" for a in AIRLINES})
    wide_prev["obs_date"] = wide_prev["obs_date"] + pd.Timedelta(days=1)

    model_df = long_df.merge(wide_prev, on=["flight_date", "obs_date"], how="inner")

    feature_cols = (
        ["airline", "days_before_departure"]
        + [f"prev_price_{a}" for a in AIRLINES]
        + [f"prev_share_{a}" for a in AIRLINES]
    )
    target_col = "price"

    model_df = model_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"학습에 쓸 행 수: {len(model_df):,}")

    train_df, test_df = flight_split(model_df)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    preprocess = ColumnTransformer(
        transformers=[("airline_ohe", OneHotEncoder(handle_unknown="ignore"), ["airline"])],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    print("=== Train ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):,.0f}")
    print(f"R2  : {r2_score(y_train, train_pred):.4f}")
    print("=== Test ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):,.0f}")
    print(f"R2  : {r2_score(y_test, test_pred):.4f}")

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
