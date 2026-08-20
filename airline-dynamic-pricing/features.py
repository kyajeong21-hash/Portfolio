from dataclasses import dataclass, field

import pandas as pd

import db

AIRLINES = ["KE", "OZ", "7C", "TW", "LJ", "TG"]

# 노트북과 동일한 값 (학습 시 상수로 고정해서 썼던 값 그대로 사용)
MARKET_SHARE = {"KE": 0.20, "OZ": 0.19, "7C": 0.16, "TW": 0.16, "LJ": 0.15, "TG": 0.14}

FEATURE_COLUMNS = (
    ["airline", "days_before_departure"]
    + [f"prev_price_{a}" for a in AIRLINES]
    + [f"prev_share_{a}" for a in AIRLINES]
)


@dataclass
class FeatureResult:
    ok: bool
    row: pd.DataFrame | None = None
    missing_airlines: list[str] = field(default_factory=list)


def build_features(
    conn,
    target_airline: str,
    flight_date: str,
    days_before_departure: int,
) -> FeatureResult:
    """
    target_airline의 가격을, flight_date 항공편의 '출발 days_before_departure일 전'
    시점에 예측하기 위한 feature 한 줄을 만듭니다.

    각 항공사의 prev_price는 "예측 시점 하루 전(obs_date - 1)까지 실제로 관측된
    가장 최근 가격"을 사용합니다. 6개 항공사 중 하나라도 그 시점까지 한 번도
    판매 기록이 없으면(과거 가격 자체가 없으면) 예측을 거부합니다.
    """
    flight_ts = pd.Timestamp(flight_date)
    obs_date = flight_ts - pd.Timedelta(days=days_before_departure)
    prev_obs_date = obs_date - pd.Timedelta(days=1)

    prev_prices = {}
    missing = []
    for airline in AIRLINES:
        price = db.get_latest_price_as_of(
            conn,
            flight_date=flight_ts.strftime("%Y-%m-%d"),
            airline=airline,
            as_of_obs_date=prev_obs_date.strftime("%Y-%m-%d"),
        )
        if price is None:
            missing.append(airline)
        else:
            prev_prices[airline] = price

    if missing:
        return FeatureResult(ok=False, missing_airlines=missing)

    data = {
        "airline": [target_airline],
        "days_before_departure": [days_before_departure],
    }
    for a in AIRLINES:
        data[f"prev_price_{a}"] = [prev_prices[a]]
    for a in AIRLINES:
        data[f"prev_share_{a}"] = [MARKET_SHARE[a]]

    row = pd.DataFrame(data)[FEATURE_COLUMNS]
    return FeatureResult(ok=True, row=row)


def build_backtest_features(
    conn,
    target_airline: str,
    flight_date: str,
    up_to_obs_date: str,
) -> pd.DataFrame:
    """
    target_airline에 대해, flight_date 항공편의 관측 이력 전체에 걸쳐
    "그 하루 전까지의 가격만 사용했다면 모델이 뭐라고 예측했을지"를
    매일 시점마다 만듭니다 (실측 라인과 비교할 예측 라인을 그리기 위한 백테스트용).

    build_features()를 날짜 수만큼 반복 호출하지 않고, 6개 항공사 가격을
    한 번에 읽어 pandas에서 ffill(과거 방향으로만)해서 만듭니다 — 결과는
    build_features()를 각 날짜에 대해 호출한 것과 동일합니다.

    반환: ["obs_date"] + FEATURE_COLUMNS. 6개 항공사 전부 과거 가격이
    있는 날짜만 포함되고(없으면 그 날짜는 건너뜀), 예측 대상 가격 자체는
    아직 계산 전입니다 (model_service.predict_batch로 넘겨서 계산).
    """
    hist = db.get_price_history(conn, flight_date, AIRLINES, up_to_obs_date)
    if hist.empty:
        return pd.DataFrame(columns=["obs_date"] + FEATURE_COLUMNS)

    wide = hist.pivot_table(index="obs_date", columns="airline", values="price")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(columns=AIRLINES)

    flight_ts = pd.Timestamp(flight_date)
    tomorrow_ts = pd.Timestamp(up_to_obs_date) + pd.Timedelta(days=1)

    full_range = pd.date_range(wide.index.min(), min(tomorrow_ts, flight_ts), freq="D")
    # 과거 방향으로만 채움 (미래 값으로 채우는 bfill은 서빙 시점엔 쓸 수 없음)
    wide = wide.reindex(full_range).ffill()

    prev = wide.shift(1)  # D일자를 예측할 때 쓰는 "D-1일자까지의 가격"
    valid_dates = prev.index[prev[AIRLINES].notna().all(axis=1)]
    if len(valid_dates) == 0:
        return pd.DataFrame(columns=["obs_date"] + FEATURE_COLUMNS)

    out = prev.loc[valid_dates, AIRLINES].copy()
    out.columns = [f"prev_price_{a}" for a in AIRLINES]
    out["airline"] = target_airline
    out["days_before_departure"] = [(flight_ts - d).days for d in valid_dates]
    for a in AIRLINES:
        out[f"prev_share_{a}"] = MARKET_SHARE[a]

    out = out.reset_index(names="obs_date")
    out["obs_date"] = out["obs_date"].dt.strftime("%Y-%m-%d")
    return out[["obs_date"] + FEATURE_COLUMNS]
