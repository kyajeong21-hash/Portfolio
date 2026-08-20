import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "flight_price.db"

AIRLINES = ["KE", "OZ", "7C", "TW", "LJ", "TG"]

# 항공사별 기본 가격대(원)
BASE_PRICE = {"KE": 550_000, "OZ": 520_000, "7C": 430_000, "TW": 440_000, "LJ": 420_000, "TG": 400_000}
# 항공사별 평균 판매 시작 리드타임(출발 며칠 전). 표준편차를 작게 둬서
# "오늘"이 출발 14일 전이어도 6개 항공사 데이터가 다 갖춰진 구간이 충분히 길게 나오도록 함
LEAD_TIME_MEAN = {"KE": 200, "OZ": 190, "7C": 180, "TW": 220, "LJ": 170, "TG": 175}
LEAD_TIME_STD = 15

N_FLIGHTS = 180
START_FLIGHT_DATE = pd.Timestamp("2030-01-01")
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    flight_dates = [START_FLIGHT_DATE + pd.Timedelta(days=i) for i in range(N_FLIGHTS)]

    for flight_date in flight_dates:
        for airline in AIRLINES:
            lead = max(20, int(rng.normal(LEAD_TIME_MEAN[airline], LEAD_TIME_STD)))
            first_obs = flight_date - pd.Timedelta(days=lead)
            obs_dates = pd.date_range(first_obs, flight_date - pd.Timedelta(days=1), freq="D")
            if len(obs_dates) == 0:
                continue

            base = BASE_PRICE[airline]
            n = len(obs_dates)
            # 완만한 랜덤워크 + 출발 임박할수록 살짝 오르는 추세 + 노이즈
            walk = rng.normal(0, 4000, size=n).cumsum()
            trend = np.linspace(0, base * 0.15, n)
            noise = rng.normal(0, 3000, size=n)
            prices = np.clip(base + walk + trend + noise, base * 0.6, base * 1.8)

            for obs_date, price in zip(obs_dates, prices):
                rows.append(
                    {
                        "flight_date": flight_date.strftime("%Y-%m-%d"),
                        "obs_date": obs_date.strftime("%Y-%m-%d"),
                        "days_before_departure": (flight_date - obs_date).days,
                        "airline": airline,
                        "price": round(float(price), -2),  # 100원 단위 반올림
                    }
                )

    df = pd.DataFrame(rows)

    con = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("flight_price", con, if_exists="replace", index=False)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_flight_price_lookup ON flight_price(flight_date, airline, obs_date)"
        )
        con.commit()
    finally:
        con.close()

    print(f"DB 저장 완료: {DB_PATH} ({len(df):,}행, 항공편 {len(flight_dates)}개)")


if __name__ == "__main__":
    main()
