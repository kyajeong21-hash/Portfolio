from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import db
import features
import model_service

DB_PATH = Path(__file__).resolve().parent / "flight_price.db"

# dataviz 팔레트 기준 항공사별 고정 색상 (red 제외, 고정 순서 유지 — 선택 항공사가
# 바뀌어도 같은 항공사는 항상 같은 색을 씀)
AIRLINE_COLORS = {
    "KE": "#2a78d6",  # blue
    "OZ": "#eb6834",  # orange
    "7C": "#1baf7a",  # aqua
    "TW": "#eda100",  # yellow
    "LJ": "#e87ba4",  # magenta
    "TG": "#008300",  # green
}
COLOR_REF_AVG = "#898781"  # 레퍼런스 '평균'은 특정 항공사가 아니므로 무채색

# 오차/방향 표시 전용 색상 (항공사 식별색과는 다른 목적의 diverging 쌍)
COLOR_OVER_PREDICT = "#e34948"  # 예측 > 실측 (과대예측)
COLOR_UNDER_PREDICT = "#2a78d6"  # 예측 < 실측 (과소예측)

EN_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def format_date(d, lang: str) -> str:
    """ko: 2023-11-01 -> 2023년 11월 1일 / en: -> November 1, 2023"""
    ts = pd.Timestamp(d)
    if lang == "ko":
        return f"{ts.year}년 {ts.month}월 {ts.day}일"
    return f"{EN_MONTHS[ts.month - 1]} {ts.day}, {ts.year}"


def format_krw_manwon(amount) -> str:
    """18930 -> '1만 9천원', -5000 -> '-5천원' (천 단위 반올림)"""
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    man, remainder = divmod(round(amount / 1000) * 1000, 10000)
    cheon = remainder // 1000
    parts = []
    if man:
        parts.append(f"{int(man)}만")
    if cheon or not parts:
        parts.append(f"{int(cheon)}천")
    return f"{sign}{' '.join(parts)}원"


def format_won(amount, lang: str) -> str:
    return f"{amount:,.0f}원" if lang == "ko" else f"{amount:,.0f} KRW"


def format_error_amount(amount, lang: str) -> str:
    if lang == "ko":
        return format_krw_manwon(amount)
    sign = "+" if amount >= 0 else ""
    return f"{sign}{amount:,.0f} KRW"


# --- 번역 테이블 ---
T = {
    "ko": {
        "app_title": "✈️ 항공권 동적 가격 예측 (프로토타입)",
        "app_subtitle": "1-period-ahead 모델: '오늘'까지의 가격으로 '내일' 가격만 예측할 수 있습니다.",
        "err_no_db": "DB 파일이 없습니다. 터미널에서 먼저 `python build_db.py`를 실행해주세요.",
        "err_no_model": "모델 파일이 없습니다. `{name}`을 이 폴더({parent})에 넣어주세요.",
        "err_no_data": "DB에 데이터가 없습니다. build_db.py가 정상적으로 실행됐는지 확인해주세요.",
        "sidebar_header": "입력",
        "sidebar_synthetic_notice": "공개용 프로토타입이라 합성 데이터를 사용합니다",
        "target_airline_label": "타겟 항공사 (본인 항공사)",
        "flight_date_label": "타겟 날짜 (출발일)",
        "reference_airlines_label": "레퍼런스 항공사 (비교하고 싶은 경쟁 항공사)",
        "today_label": "오늘 날짜 (기준일)",
        "airline_colors_caption": "항공사 색상",
        "err_date_after_departure": "오늘이 이미 출발일이거나 그 이후라 예측할 수 없습니다. 다른 날짜를 선택해주세요.",
        "predict_target_caption": "→ **{date}** (D-{d}) 가격을 예측합니다.",
        "warn_no_prediction": (
            "예측할 수 없습니다: 아래 항공사가 오늘까지 판매 기록이 없어 예측 근거가 부족합니다.\n\n"
            "- 판매 기록 없는 항공사: {airlines}"
        ),
        "success_prediction": "{airline} {date} 예측 가격: **{price}**",
        "change_no_yesterday": "전일 실측가 없음",
        "change_none": "변화 없음",
        "hero_today_label": "오늘 가격",
        "hero_tomorrow_label": "내일 예측가격",
        "hero_mae_label": "예상 오차 범위 · MAE 기준",
        "info_no_hist": "표시할 가격 이력이 없습니다.",
        "chart1_subheader": "① 타겟 실측가 VS 예측가",
        "info_no_target_hist": "타겟 항공사의 가격 이력이 없습니다.",
        "actual_price_label": "실측가",
        "predicted_price_label": "예측가",
        "x_axis_sales_date": "판매일",
        "y_axis_price": "가격(원)",
        "flight_caption": "출발편: {date}",
        "error_analysis_header": "📊 예측 오차 분석",
        "info_no_predicted_hist": "오차를 계산할 예측 이력이 없습니다.",
        "info_no_overlap": "실측/예측이 겹치는 날짜가 없어 오차를 계산할 수 없습니다.",
        "mae_metric": "MAE (평균절대오차)",
        "rmse_metric": "RMSE",
        "mape_metric": "MAPE (평균절대오차율)",
        "bias_metric": "Bias (평균오차)",
        "bias_over": "과대예측 경향",
        "bias_under": "과소예측 경향",
        "error_chart_y_axis": "오차",
        "error_legend_caption": "🔴 과대예측(예측 > 실측)  🔵 과소예측(예측 < 실측) · n={n}일 기준",
        "top5_worst": "오차 가장 컸던 날 TOP 5",
        "top5_best": "오차 가장 작았던 날 TOP 5",
        "col_sales_date": "판매일",
        "col_actual": "실측가",
        "col_predicted": "예측가",
        "col_error": "오차",
        "col_error_pct": "오차율(%)",
        "chart2_subheader": "② 타겟 vs 레퍼런스 평균",
        "series_target_suffix": "(타겟)",
        "series_ref_avg": "레퍼런스 평균",
        "dash_actual": "실측",
        "dash_predicted": "예측",
        "chart3_subheader": "③ 항공사별 가격 비교",
        "visible_airlines_label": "표시할 항공사 선택",
        "tooltip_forecast_price": "예측가",
        "col_airline": "항공사",
        "chart_hint": "🔍 드래그/스크롤로 확대·축소할 수 있고, 점에 마우스를 올리면 세부 정보가 표시됩니다",
    },
    "en": {
        "app_title": "✈️ Airline Dynamic Price Prediction (Prototype)",
        "app_subtitle": "1-period-ahead model: predicts only 'tomorrow's' price using data up through 'today'.",
        "err_no_db": "DB file not found. Please run `python build_db.py` in the terminal first.",
        "err_no_model": "Model file not found. Please place `{name}` in this folder ({parent}).",
        "err_no_data": "No data in the DB. Please check that build_db.py ran successfully.",
        "sidebar_header": "Input",
        "sidebar_synthetic_notice": "This is a public prototype, so it uses synthetic data",
        "target_airline_label": "Target airline (your airline)",
        "flight_date_label": "Target date (departure)",
        "reference_airlines_label": "Reference airlines (competitors to compare)",
        "today_label": "Today's date (reference point)",
        "airline_colors_caption": "Airline colors",
        "err_date_after_departure": "Today is already the departure date or later, so prediction isn't possible. Please choose a different date.",
        "predict_target_caption": "→ Predicting the price for **{date}** (D-{d}).",
        "warn_no_prediction": (
            "Prediction not possible: the airline(s) below have no sales record up to today, "
            "so there isn't enough basis to predict.\n\n"
            "- Airlines with no sales record: {airlines}"
        ),
        "success_prediction": "{airline} {date} predicted price: **{price}**",
        "change_no_yesterday": "No actual price for yesterday",
        "change_none": "No change",
        "hero_today_label": "Today's price",
        "hero_tomorrow_label": "Tomorrow's predicted price",
        "hero_mae_label": "Expected error range · based on MAE",
        "info_no_hist": "No price history to display.",
        "chart1_subheader": "① Target: Actual vs Predicted",
        "info_no_target_hist": "No price history for the target airline.",
        "actual_price_label": "Actual price",
        "predicted_price_label": "Predicted price",
        "x_axis_sales_date": "Sales date",
        "y_axis_price": "Price (KRW)",
        "flight_caption": "Departure flight: {date}",
        "error_analysis_header": "📊 Prediction Error Analysis",
        "info_no_predicted_hist": "No prediction history available to compute error.",
        "info_no_overlap": "No overlapping dates between actual and predicted, so error can't be computed.",
        "mae_metric": "MAE (Mean Absolute Error)",
        "rmse_metric": "RMSE",
        "mape_metric": "MAPE (Mean Absolute % Error)",
        "bias_metric": "Bias (Mean Error)",
        "bias_over": "Tends to over-predict",
        "bias_under": "Tends to under-predict",
        "error_chart_y_axis": "Error",
        "error_legend_caption": "🔴 Over-prediction (pred > actual)  🔵 Under-prediction (pred < actual) · based on n={n} days",
        "top5_worst": "Top 5 Largest Errors",
        "top5_best": "Top 5 Smallest Errors",
        "col_sales_date": "Sales date",
        "col_actual": "Actual",
        "col_predicted": "Predicted",
        "col_error": "Error",
        "col_error_pct": "Error %",
        "chart2_subheader": "② Target vs Reference Average",
        "series_target_suffix": "(Target)",
        "series_ref_avg": "Reference average",
        "dash_actual": "Actual",
        "dash_predicted": "Predicted",
        "chart3_subheader": "③ Price Comparison by Airline",
        "visible_airlines_label": "Select airlines to display",
        "tooltip_forecast_price": "Predicted price",
        "col_airline": "Airline",
        "chart_hint": "🔍 Drag/scroll to zoom, hover over a point for details",
    },
}

st.set_page_config(page_title="항공권 가격 예측 / Price Prediction", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 260px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

lang_choice = st.sidebar.selectbox("🌐 Language / 언어", ["한국어", "English"])
lang = "ko" if lang_choice == "한국어" else "en"


def t(key, **kwargs):
    text = T[lang][key]
    return text.format(**kwargs) if kwargs else text


def chart_hint():
    st.markdown(
        f'<div style="font-size:11px; color:#898781; margin-top:-8px; margin-bottom:20px;">{t("chart_hint")}</div>',
        unsafe_allow_html=True,
    )


# 날짜 범위가 좁으면 Vega-Lite가 자동으로 시/분 단위 눈금(요일+시각)을 쓰거나
# 로케일 기본값(영어)으로 표시하는 문제가 있어, 포맷을 날짜 형식으로 고정합니다.
DATE_FORMAT = "%Y-%m-%d"


def date_axis():
    return alt.Axis(title=t("x_axis_sales_date"), format=DATE_FORMAT)


def date_tooltip():
    return alt.Tooltip("obs_date:T", title=t("x_axis_sales_date"), format=DATE_FORMAT)


st.header(t("app_title"))
st.caption(t("app_subtitle"))
st.markdown(
    f'<div style="font-size:11px; color:#898781; margin-top:-8px; margin-bottom:12px;">{t("sidebar_synthetic_notice")}</div>',
    unsafe_allow_html=True,
)

if not DB_PATH.exists():
    st.error(t("err_no_db"))
    st.stop()

if not model_service.MODEL_PATH.exists():
    st.error(t("err_no_model", name=model_service.MODEL_PATH.name, parent=model_service.MODEL_PATH.parent))
    st.stop()

conn = db.get_connection()
flight_dates = db.get_flight_dates(conn)
if not flight_dates:
    st.error(t("err_no_data"))
    st.stop()

with st.sidebar:
    st.header(t("sidebar_header"))
    target_airline = st.selectbox(t("target_airline_label"), features.AIRLINES)
    flight_date = st.selectbox(
        t("flight_date_label"), flight_dates, format_func=lambda d: format_date(d, lang)
    )

    reference_options = [a for a in features.AIRLINES if a != target_airline]
    reference_airlines = st.multiselect(
        t("reference_airlines_label"),
        reference_options,
        default=reference_options,
    )

    default_today = (pd.Timestamp(flight_date) - pd.Timedelta(days=14)).date()
    today = st.date_input(t("today_label"), value=default_today)

    st.divider()
    st.caption(t("airline_colors_caption"))
    for a in features.AIRLINES:
        st.markdown(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{AIRLINE_COLORS[a]};margin-right:6px;"></span>{a}',
            unsafe_allow_html=True,
        )

flight_ts = pd.Timestamp(flight_date)
today_ts = pd.Timestamp(today)
tomorrow_ts = today_ts + pd.Timedelta(days=1)
days_before_departure = (flight_ts - tomorrow_ts).days

if days_before_departure < 0:
    st.error(t("err_date_after_departure"))
    st.stop()

st.caption(t("predict_target_caption", date=format_date(tomorrow_ts, lang), d=days_before_departure))


# --- 예측 ---
result = features.build_features(
    conn,
    target_airline=target_airline,
    flight_date=flight_date,
    days_before_departure=days_before_departure,
)

forecast_price = None
if not result.ok:
    st.warning(t("warn_no_prediction", airlines=", ".join(result.missing_airlines)))
else:
    forecast_price = model_service.predict_price(result.row)

# --- 그래프 데이터 준비 (3개 차트 공통) ---
today_str = today_ts.strftime("%Y-%m-%d")
target_hist = db.get_price_history(conn, flight_date, [target_airline], today_str)
ref_hist = db.get_price_history(conn, flight_date, reference_airlines, today_str)

target_label = f"{target_airline} {t('series_target_suffix')}"
target_color = AIRLINE_COLORS[target_airline]

DASH_SCALE = alt.Scale(domain=[t("dash_actual"), t("dash_predicted")], range=[[1, 0], [6, 4]])
DASH_SCALE_LABELED = alt.Scale(
    domain=[t("actual_price_label"), t("predicted_price_label")], range=[[1, 0], [6, 4]]
)

backtest_df = features.build_backtest_features(conn, target_airline, flight_date, today_str)
predicted_df = None
if not backtest_df.empty:
    preds = model_service.predict_batch(backtest_df[features.FEATURE_COLUMNS])
    predicted_df = pd.DataFrame({"obs_date": backtest_df["obs_date"], "price": preds})

today_actual = target_hist.iloc[-1]["price"] if not target_hist.empty else None

# --- 백테스트 오차 지표 (히어로 카드 + 아래 오차 분석 대시보드에서 공용으로 사용) ---
error_df = None
mae = rmse = mape = bias = None
if predicted_df is not None and not predicted_df.empty and not target_hist.empty:
    error_df = pd.merge(
        target_hist[["obs_date", "price"]].rename(columns={"price": "actual"}),
        predicted_df.rename(columns={"price": "predicted"}),
        on="obs_date",
        how="inner",
    )
    if not error_df.empty:
        error_df["error"] = error_df["predicted"] - error_df["actual"]
        error_df["abs_error"] = error_df["error"].abs()
        error_df["error_pct"] = error_df["error"] / error_df["actual"] * 100
        mae = error_df["abs_error"].mean()
        rmse = (error_df["error"] ** 2).mean() ** 0.5
        mape = error_df["error_pct"].abs().mean()
        bias = error_df["error"].mean()

# --- 오늘 가격 / 내일 예측가격 / 오차 범위 (큰 숫자로 표시) ---
if forecast_price is not None:
    st.success(
        t(
            "success_prediction",
            airline=target_airline,
            date=format_date(tomorrow_ts, lang),
            price=format_won(forecast_price, lang),
        )
    )

    change = (forecast_price - today_actual) if today_actual is not None else None
    change_pct = (change / today_actual * 100) if change is not None else None
    if change is None:
        change_color, change_text = "#52514e", t("change_no_yesterday")
    elif change > 0:
        change_color, change_text = COLOR_OVER_PREDICT, f"▲ {format_error_amount(change, lang)} ({change_pct:+.1f}%)"
    elif change < 0:
        change_color, change_text = COLOR_UNDER_PREDICT, f"▼ {format_error_amount(change, lang)} ({change_pct:+.1f}%)"
    else:
        change_color, change_text = "#52514e", t("change_none")

    hc1, hc2, hc3 = st.columns([1, 1.4, 1])
    with hc1:
        today_text = format_won(today_actual, lang) if today_actual is not None else "N/A"
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e1e0d9; border-radius:12px; padding:18px; height:100%;">
              <div style="font-size:13px; color:#52514e;">{t("hero_today_label")} · {format_date(today_ts, lang)}</div>
              <div style="font-size:38px; font-weight:700; color:#0b0b0b;">{today_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hc2:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e1e0d9; border-radius:12px; padding:18px; height:100%;">
              <div style="font-size:13px; color:#52514e;">{t("hero_tomorrow_label")} · {format_date(tomorrow_ts, lang)}</div>
              <div style="font-size:38px; font-weight:700; color:{target_color};">{format_won(forecast_price, lang)}
                <span style="font-size:14px; font-weight:600; color:{change_color}; margin-left:8px;">{change_text}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hc3:
        mae_text = f"± {format_won(mae, lang)}" if mae is not None else "N/A"
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e1e0d9; border-radius:12px; padding:18px; height:100%;">
              <div style="font-size:13px; color:#52514e;">{t("hero_mae_label")}</div>
              <div style="font-size:38px; font-weight:700; color:#0b0b0b;">{mae_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")


def forecast_point_layer(color):
    """내일자 예측치 1점을 다이아몬드로 강조 (범례 없는 보조 레이어)."""
    if predicted_df is None or predicted_df.empty:
        return []
    return [
        alt.Chart(predicted_df.tail(1))
        .mark_point(shape="diamond", size=120, filled=True, color=color)
        .encode(
            x=alt.X("obs_date:T", axis=date_axis()),
            y="price:Q",
            tooltip=[date_tooltip(), alt.Tooltip("price:Q", format=",.0f", title=t("tooltip_forecast_price"))],
        )
    ]


if target_hist.empty and ref_hist.empty:
    st.info(t("info_no_hist"))
else:
    # === ① 타겟 실측가 VS 예측가 (축 확대) ===
    st.subheader(t("chart1_subheader"))
    if target_hist.empty:
        st.info(t("info_no_target_hist"))
    else:
        rows1 = [
            {"obs_date": r.obs_date, "price": r.price, "kind": t("actual_price_label")}
            for r in target_hist.itertuples()
        ]
        if predicted_df is not None and not predicted_df.empty:
            rows1 += [
                {"obs_date": r.obs_date, "price": r.price, "kind": t("predicted_price_label")}
                for r in predicted_df.itertuples()
            ]
        chart1_df = pd.DataFrame(rows1)

        TICK_STEP = 25_000
        vmin, vmax = chart1_df["price"].min(), chart1_df["price"].max()
        pad = max((vmax - vmin) * 0.1, 1)
        y_domain1 = [vmin - pad, vmax + pad]
        tick_start = (y_domain1[0] // TICK_STEP) * TICK_STEP
        tick_end = (y_domain1[1] // TICK_STEP + 1) * TICK_STEP
        tick_values1 = list(range(int(tick_start), int(tick_end) + 1, TICK_STEP))

        line1 = (
            alt.Chart(chart1_df)
            .mark_line(strokeWidth=2, color=target_color, point=alt.OverlayMarkDef(size=40, color=target_color))
            .encode(
                x=alt.X("obs_date:T", axis=date_axis()),
                y=alt.Y(
                    "price:Q",
                    title=t("y_axis_price"),
                    scale=alt.Scale(domain=y_domain1, zero=False),
                    axis=alt.Axis(values=tick_values1),
                ),
                strokeDash=alt.StrokeDash("kind:N", scale=DASH_SCALE_LABELED, title=None),
                tooltip=[date_tooltip(), "kind:N", alt.Tooltip("price:Q", format=",.0f")],
            )
        )
        st.caption(t("flight_caption", date=format_date(flight_date, lang)))
        st.altair_chart(
            alt.layer(line1, *forecast_point_layer(target_color)).properties(height=340).interactive(),
            width="stretch",
        )
        chart_hint()

        # --- 예측 오차 분석 대시보드 ---
        st.markdown(f"##### {t('error_analysis_header')}")
        if predicted_df is None or predicted_df.empty:
            st.info(t("info_no_predicted_hist"))
        elif error_df is None or error_df.empty:
            st.info(t("info_no_overlap"))
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(t("mae_metric"), format_won(mae, lang))
            m2.metric(t("rmse_metric"), format_won(rmse, lang))
            m3.metric(t("mape_metric"), f"{mape:.2f}%")
            m4.metric(
                t("bias_metric"),
                format_error_amount(bias, lang),
                delta=t("bias_over") if bias > 0 else t("bias_under"),
                delta_color="off",
            )

            error_chart = (
                alt.Chart(error_df)
                .mark_bar()
                .encode(
                    x=alt.X("obs_date:T", axis=date_axis()),
                    y=alt.Y("error:Q", title=t("error_chart_y_axis")),
                    color=alt.condition(
                        alt.datum["error"] > 0,
                        alt.value(COLOR_OVER_PREDICT),
                        alt.value(COLOR_UNDER_PREDICT),
                    ),
                    tooltip=[
                        date_tooltip(),
                        alt.Tooltip("actual:Q", format=",.0f", title=t("actual_price_label")),
                        alt.Tooltip("predicted:Q", format=",.0f", title=t("predicted_price_label")),
                        alt.Tooltip("error:Q", format=",.0f", title=t("col_error")),
                    ],
                )
                .properties(height=180)
                .interactive()
            )
            st.altair_chart(error_chart, width="stretch")
            st.caption(t("error_legend_caption", n=len(error_df)))

            display_df = error_df.copy()
            display_df["error"] = display_df["error"].apply(lambda v: format_error_amount(v, lang))
            display_df["actual"] = display_df["actual"].apply(lambda v: format_won(v, lang))
            display_df["predicted"] = display_df["predicted"].apply(lambda v: format_won(v, lang))
            display_df["error_pct"] = display_df["error_pct"].apply(lambda v: f"{v:+.1f}%")
            display_df["obs_date"] = display_df["obs_date"].apply(lambda d: format_date(d, lang))
            display_df = display_df.rename(
                columns={
                    "obs_date": t("col_sales_date"),
                    "actual": t("col_actual"),
                    "predicted": t("col_predicted"),
                    "error": t("col_error"),
                    "error_pct": t("col_error_pct"),
                }
            )
            display_cols = [t("col_sales_date"), t("col_actual"), t("col_predicted"), t("col_error"), t("col_error_pct")]

            worst_order = error_df["abs_error"].sort_values(ascending=False).index
            best_order = error_df["abs_error"].sort_values(ascending=True).index

            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"**{t('top5_worst')}**")
                st.dataframe(display_df.loc[worst_order].head(5)[display_cols], hide_index=True)
            with t2:
                st.markdown(f"**{t('top5_best')}**")
                st.dataframe(display_df.loc[best_order].head(5)[display_cols], hide_index=True)

    st.divider()

    # === ② 타겟 vs 레퍼런스 평균 ===
    st.subheader(t("chart2_subheader"))
    rows2 = [
        {"obs_date": r.obs_date, "price": r.price, "series": target_label, "kind": t("dash_actual")}
        for r in target_hist.itertuples()
    ]
    if not ref_hist.empty:
        ref_avg = ref_hist.groupby("obs_date", as_index=False)["price"].mean()
        rows2 += [
            {"obs_date": r.obs_date, "price": r.price, "series": t("series_ref_avg"), "kind": t("dash_actual")}
            for r in ref_avg.itertuples()
        ]
    if predicted_df is not None and not predicted_df.empty:
        rows2 += [
            {"obs_date": r.obs_date, "price": r.price, "series": target_label, "kind": t("dash_predicted")}
            for r in predicted_df.itertuples()
        ]
    chart2_df = pd.DataFrame(rows2)
    color_scale2 = alt.Scale(domain=[target_label, t("series_ref_avg")], range=[target_color, COLOR_REF_AVG])
    line2 = (
        alt.Chart(chart2_df)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40))
        .encode(
            x=alt.X("obs_date:T", axis=date_axis()),
            y=alt.Y("price:Q", title=t("y_axis_price"), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=color_scale2, title=None),
            strokeDash=alt.StrokeDash("kind:N", scale=DASH_SCALE, legend=None),
            tooltip=[date_tooltip(), "series:N", "kind:N", alt.Tooltip("price:Q", format=",.0f")],
        )
    )
    st.caption(t("flight_caption", date=format_date(flight_date, lang)))
    st.altair_chart(
        alt.layer(line2, *forecast_point_layer(target_color)).properties(height=340).interactive(),
        width="stretch",
    )
    chart_hint()

    # === ③ 항공사별 가격 비교 (개별, 평균 아님) ===
    st.subheader(t("chart3_subheader"))
    shown_airlines = [a for a in features.AIRLINES if a == target_airline or a in reference_airlines]
    rows3 = [
        {"obs_date": r.obs_date, "price": r.price, "airline": r.airline, "kind": t("dash_actual")}
        for r in target_hist.itertuples()
    ]
    rows3 += [
        {"obs_date": r.obs_date, "price": r.price, "airline": r.airline, "kind": t("dash_actual")}
        for r in ref_hist.itertuples()
    ]
    if predicted_df is not None and not predicted_df.empty:
        rows3 += [
            {"obs_date": r.obs_date, "price": r.price, "airline": target_airline, "kind": t("dash_predicted")}
            for r in predicted_df.itertuples()
        ]
    chart3_df = pd.DataFrame(rows3)

    if chart3_df.empty:
        st.info(t("info_no_hist"))
    else:
        visible_airlines = st.multiselect(
            t("visible_airlines_label"),
            shown_airlines,
            default=shown_airlines,
            key="chart3_visible_airlines",
        )
        chart3_visible_df = chart3_df[chart3_df["airline"].isin(visible_airlines)]

        color_scale3 = alt.Scale(domain=shown_airlines, range=[AIRLINE_COLORS[a] for a in shown_airlines])
        line3 = (
            alt.Chart(chart3_visible_df)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
            .encode(
                x=alt.X("obs_date:T", axis=date_axis()),
                y=alt.Y("price:Q", title=t("y_axis_price"), scale=alt.Scale(zero=False)),
                color=alt.Color("airline:N", scale=color_scale3, title=None),
                strokeDash=alt.StrokeDash("kind:N", scale=DASH_SCALE, legend=None),
                tooltip=[date_tooltip(), "airline:N", "kind:N", alt.Tooltip("price:Q", format=",.0f")],
            )
        )
        st.caption(t("flight_caption", date=format_date(flight_date, lang)))
        st.altair_chart(
            alt.layer(line3, *forecast_point_layer(target_color)).properties(height=380).interactive(),
            width="stretch",
        )
        chart_hint()

conn.close()
