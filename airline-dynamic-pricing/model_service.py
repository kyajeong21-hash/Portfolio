"""
모델 로딩 & 예측 래퍼.

앱이 켜질 때 딱 한 번만 pkl을 로드해서 재사용합니다.
(요청마다 joblib.load를 부르면 매번 100MB짜리 파일을 다시 읽어서 느려집니다.)
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "Final_Model.pkl"


@lru_cache(maxsize=1)
def get_pipeline():
    return joblib.load(MODEL_PATH)


def predict_price(feature_row: pd.DataFrame) -> float:
    pipeline = get_pipeline()
    pred = pipeline.predict(feature_row)
    return float(pred[0])


def predict_batch(feature_rows: pd.DataFrame):
    """여러 행을 한 번에 예측 (백테스트용 예측 라인 등). 행 개수만큼의 배열 반환."""
    pipeline = get_pipeline()
    return pipeline.predict(feature_rows)
