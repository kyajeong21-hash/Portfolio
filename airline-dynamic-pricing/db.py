import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "flight_price.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_flight_dates(conn) -> list[str]:
    """프로토타입 UI에서 선택 가능한 flight_date 목록."""
    df = pd.read_sql(
        "SELECT DISTINCT flight_date FROM flight_price ORDER BY flight_date", conn
    )
    return df["flight_date"].tolist()


def get_latest_price_as_of(conn, flight_date: str, airline: str, as_of_obs_date: str):
    """
    특정 항공편(flight_date)의 특정 항공사(airline) 가격 중,
    as_of_obs_date '이전(포함)' 시점에 실제로 관측된 가장 최근 가격 1건을 가져옵니다.

    -- 학습 코드의 ffill과 동일한 의미: "최근에 팔린 적 있으면 그 가격을 그대로 씀".
    -- 미래 시점 가격은 절대 보지 않습니다 (bfill 미적용, 실시간 서빙이므로).

    결과가 없으면 None (= 그 시점까지 한 번도 판매 기록이 없다는 뜻).
    """
    query = """
        SELECT obs_date, price
        FROM flight_price
        WHERE flight_date = ?
          AND airline = ?
          AND obs_date <= ?
        ORDER BY obs_date DESC
        LIMIT 1
    """
    df = pd.read_sql(query, conn, params=[flight_date, airline, as_of_obs_date])
    if df.empty:
        return None
    return float(df.iloc[0]["price"])


def get_price_history(conn, flight_date: str, airlines: list[str], up_to_obs_date: str) -> pd.DataFrame:
    """
    해당 flight_date의, airlines에 해당하는 항공사들의 '실제 관측된' 가격을
    obs_date <= up_to_obs_date 범위에서 그대로 가져옵니다 (결측 보정 없음 —
    그래프에서는 실제로 관측된 값만 점으로 찍기 위함).
    """
    if not airlines:
        return pd.DataFrame(columns=["obs_date", "airline", "price"])

    placeholders = ",".join("?" for _ in airlines)
    query = f"""
        SELECT obs_date, airline, price
        FROM flight_price
        WHERE flight_date = ?
          AND airline IN ({placeholders})
          AND obs_date <= ?
        ORDER BY obs_date
    """
    params = [flight_date, *airlines, up_to_obs_date]
    return pd.read_sql(query, conn, params=params)
