#!/usr/bin/env python3
"""
Market Intelligence for ChatGPT trading analysis.

This script is deliberately isolated from the existing stock screener.

It DOES NOT change:
- latest.json
- latest_eod.json
- latest_intraday.json
- screener thresholds
- Telegram alerts

Outputs:

EOD:
    watchlists/latest_market.json
    watchlists/latest_universe.json

Intraday:
    watchlists/latest_market.json
    watchlists/latest_intraday_universe.json

The intraday universe contains ALL valid stocks with available 5-minute data,
not only stocks that triggered EP / breakout signals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
import yfinance as yf


# ============================================================
# Time zones
# ============================================================

NY = ZoneInfo("America/New_York")
HK = ZoneInfo("Asia/Hong_Kong")
UTC = ZoneInfo("UTC")


# ============================================================
# Paths
# ============================================================

WATCHLIST_DIR = Path("watchlists")

MARKET_PATH = (
    WATCHLIST_DIR
    / "latest_market.json"
)

UNIVERSE_PATH = (
    WATCHLIST_DIR
    / "latest_universe.json"
)

INTRADAY_UNIVERSE_PATH = (
    WATCHLIST_DIR
    / "latest_intraday_universe.json"
)


# ============================================================
# Config
# ============================================================

MIN_DAILY_COVERAGE = 0.80
MIN_INTRADAY_COVERAGE = 0.65

DAILY_BATCH_SIZE = 45
INTRADAY_BATCH_SIZE = 45

MARKET_SESSION_MINUTES = 390

NEAR_BREAKOUT_DISTANCE_PCT = 1.0
HIGH_RVOL_THRESHOLD = 2.0


INVALID_TICKERS = {
    "INFO",
    "ATVI",
    "ALXN",
    "CELG",
    "CTXS",
    "MXIM",
    "NUAN",
    "ABC",
    "BLL",
    "COG",
    "DISCA",
    "DISCK",
    "DRE",
    "FISV",
    "KSU",
    "MYL",
    "NBL",
    "PEAK",
    "FB",
    "NDX",
    "ARKG",
    "INFQ",
    "SOLS",
    "IMSR",
    "SPCX",
    "EIP",
    "ILLM",
}


SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Health Care": "XLV",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


# ============================================================
# Basic helpers
# ============================================================

def clean_ticker(value):
    if value is None:
        return None

    ticker = (
        str(value)
        .upper()
        .strip()
        .split("#")[0]
        .strip()
        .replace(".", "-")
    )

    if not ticker:
        return None

    if ticker in INVALID_TICKERS:
        return None

    if not ticker.replace("-", "").isalnum():
        return None

    return ticker


def load_universe(
    config_path="config.yaml",
):
    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f) or {}

    out = []
    seen = set()

    for raw in cfg.get(
        "stock_universe",
        [],
    ):
        ticker = clean_ticker(raw)

        if (
            ticker
            and ticker not in seen
        ):
            seen.add(ticker)
            out.append(ticker)

    if not out:
        raise RuntimeError(
            "config.yaml -> "
            "stock_universe is empty"
        )

    return out


def flatten_single(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        if (
            len(
                df.columns
                .get_level_values(-1)
                .unique()
            )
            == 1
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

    return df


def extract_ticker_df(
    data,
    ticker,
    batch_len,
):
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if batch_len == 1:
            return flatten_single(
                data.copy()
            )

        if not isinstance(
            data.columns,
            pd.MultiIndex,
        ):
            return pd.DataFrame()

        level0 = set(
            map(
                str,
                data.columns
                .get_level_values(0),
            )
        )

        level_last = set(
            map(
                str,
                data.columns
                .get_level_values(-1),
            )
        )

        if ticker in level0:
            return flatten_single(
                data[ticker].copy()
            )

        if ticker in level_last:
            return flatten_single(
                data.xs(
                    ticker,
                    axis=1,
                    level=-1,
                ).copy()
            )

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def finite_or_none(
    value,
    digits=4,
):
    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return round(
        value,
        digits,
    )


def pct_change(
    current,
    previous,
):
    try:
        current = float(current)
        previous = float(previous)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if (
        not math.isfinite(current)
        or not math.isfinite(previous)
        or previous == 0
    ):
        return None

    return (
        current / previous
        - 1.0
    ) * 100.0


def close_return(
    df,
    sessions,
):
    if (
        df is None
        or df.empty
        or len(df) <= sessions
    ):
        return None

    return pct_change(
        df["Close"].iloc[-1],
        df["Close"].iloc[
            -1 - sessions
        ],
    )


def completed_daily(df):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    required = [
        c
        for c in [
            "Close",
            "High",
            "Low",
        ]
        if c in df.columns
    ]

    if len(required) < 3:
        return pd.DataFrame()

    return df.dropna(
        subset=[
            "Close",
            "High",
            "Low",
        ]
    ).copy()


def strip_current_session_daily_bar(
    df,
    session_date,
):
    """
    During intraday mode, yfinance may include
    today's unfinished daily candle.

    Remove it so:
    - prev_close
    - prev_high
    - ATR
    - moving averages

    all refer to completed sessions.
    """

    out = completed_daily(df)

    if out.empty:
        return out

    try:
        last_ts = pd.Timestamp(
            out.index[-1]
        )

        if last_ts.tzinfo is not None:
            last_date = (
                last_ts
                .tz_convert(NY)
                .date()
            )

        else:
            last_date = (
                last_ts.date()
            )

        if (
            last_date >= session_date
            and len(out) >= 2
        ):
            out = (
                out.iloc[:-1]
                .copy()
            )

    except Exception:
        pass

    return out


# ============================================================
# Downloads
# ============================================================

def download_batched(
    tickers,
    period,
    interval,
    batch_size=45,
):
    result = {}
    failures = []

    for start in range(
        0,
        len(tickers),
        batch_size,
    ):
        batch = tickers[
            start:
            start + batch_size
        ]

        print(
            f"Downloading "
            f"{interval}: "
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"of {len(tickers)}"
        )

        try:
            raw = yf.download(
                " ".join(batch),
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                auto_adjust=False,
                prepost=False,
                threads=True,
            )

        except Exception as exc:
            failures.extend(batch)

            print(
                "Download failed for "
                f"batch {start}: {exc}",
                file=sys.stderr,
            )

            continue

        for ticker in batch:
            df = extract_ticker_df(
                raw,
                ticker,
                len(batch),
            )

            if (
                df is None
                or df.empty
            ):
                failures.append(ticker)

            else:
                result[ticker] = df

    return result, failures


# ============================================================
# Indicators
# ============================================================

def atr14(df):
    """
    Wilder ATR14.

    Equivalent principle to:
    TradingView ta.atr(14)
    """

    if (
        df is None
        or len(df) < 15
    ):
        return None

    high = (
        df["High"]
        .astype(float)
    )

    low = (
        df["Low"]
        .astype(float)
    )

    close = (
        df["Close"]
        .astype(float)
    )

    previous_close = (
        close.shift(1)
    )

    true_range = pd.concat(
        [
            high - low,

            (
                high
                - previous_close
            ).abs(),

            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        )
        .mean()
    )

    value = atr.iloc[-1]

    if pd.isna(value):
        return None

    return float(value)


def adr20(df):
    if (
        df is None
        or len(df) < 20
    ):
        return None

    recent = df.iloc[-20:]

    values = (
        (
            recent["High"]
            - recent["Low"]
        )
        / recent["Close"]
        * 100.0
    )

    if values.isna().all():
        return None

    return float(
        values.mean()
    )


def avg_dollar_volume(df):
    if (
        df is None
        or len(df) < 20
        or "Volume"
        not in df.columns
    ):
        return None

    recent = df.iloc[-20:]

    values = (
        recent["Close"]
        * recent["Volume"]
    )

    if values.isna().all():
        return None

    return float(
        values.mean()
    )


def volume_dry_up_ratio(df):
    if (
        df is None
        or len(df) < 25
        or "Volume"
        not in df.columns
    ):
        return None

    recent_5 = float(
        df["Volume"]
        .iloc[-5:]
        .mean()
    )

    previous_20 = float(
        df["Volume"]
        .iloc[-25:-5]
        .mean()
    )

    if (
        not math.isfinite(
            previous_20
        )
        or previous_20 <= 0
    ):
        return None

    return (
        recent_5
        / previous_20
    )


def range_10_pct(df):
    if (
        df is None
        or len(df) < 10
    ):
        return None

    recent = df.iloc[-10:]

    low = float(
        recent["Low"].min()
    )

    high = float(
        recent["High"].max()
    )

    if low <= 0:
        return None

    return (
        (high - low)
        / low
        * 100.0
    )


def distance_52w_high(df):
    if (
        df is None
        or len(df) < 100
    ):
        return None

    recent = (
        df.iloc[-252:]
        if len(df) >= 252
        else df
    )

    high = float(
        recent["High"].max()
    )

    close = float(
        df["Close"].iloc[-1]
    )

    if high <= 0:
        return None

    return (
        (high - close)
        / high
        * 100.0
    )


def relative_strength(
    stock_df,
    benchmark_df,
    sessions,
):
    stock_ret = close_return(
        stock_df,
        sessions,
    )

    bench_ret = close_return(
        benchmark_df,
        sessions,
    )

    if (
        stock_ret is None
        or bench_ret is None
    ):
        return None

    return (
        stock_ret
        - bench_ret
    )


def extension_zone(value):
    if value is None:
        return "UNKNOWN"

    if value <= -5:
        return "EXTREME_OVERSOLD"

    if value <= -3:
        return "DEPRESSED"

    if value < 0:
        return "BELOW_50MA"

    if value < 1:
        return "NEAR_50MA"

    if value < 2:
        return "ABOVE_50MA"

    if value < 3:
        return "EXTENDED"

    if value < 5:
        return "VERY_EXTENDED"

    return "EXTREME_OVERBOUGHT"


def trend_state(
    price,
    sma20,
    sma50,
    sma200,
    slope,
):
    if any(
        value is None
        for value in (
            price,
            sma20,
            sma50,
            sma200,
        )
    ):
        return "UNKNOWN"

    if (
        price > sma20
        > sma50
        > sma200
        and slope is not None
        and slope > 0
    ):
        return "STRONG_UPTREND"

    if (
        price > sma50
        > sma200
        and slope is not None
        and slope > 0
    ):
        return "UPTREND"

    if (
        price > sma200
        and price < sma50
    ):
        return (
            "PULLBACK_ABOVE_200MA"
        )

    if (
        price < sma50
        and price < sma200
    ):
        return "DOWNTREND"

    return "MIXED"


# ============================================================
# EOD universe
# ============================================================

def build_universe_row(
    ticker,
    df,
    spy_df,
    qqq_df,
):
    df = completed_daily(df)

    if len(df) < 220:
        return None

    close = (
        df["Close"]
        .astype(float)
    )

    price = float(
        close.iloc[-1]
    )

    sma20_series = (
        close
        .rolling(20)
        .mean()
    )

    sma50_series = (
        close
        .rolling(50)
        .mean()
    )

    sma200_series = (
        close
        .rolling(200)
        .mean()
    )

    sma20 = finite_or_none(
        sma20_series.iloc[-1]
    )

    sma50 = finite_or_none(
        sma50_series.iloc[-1]
    )

    sma200 = finite_or_none(
        sma200_series.iloc[-1]
    )

    slope = None

    if (
        len(df) >= 60
        and not pd.isna(
            sma50_series.iloc[-11]
        )
        and sma50_series.iloc[-11]
        != 0
    ):
        slope = (
            (
                float(
                    sma50_series
                    .iloc[-1]
                )
                / float(
                    sma50_series
                    .iloc[-11]
                )
            )
            - 1.0
        ) * 100.0

    atr = atr14(df)

    atr_ext = None

    if (
        atr is not None
        and atr > 0
        and sma50 is not None
    ):
        atr_ext = (
            price - sma50
        ) / atr

    price_date = (
        pd.Timestamp(
            df.index[-1]
        )
        .date()
        .isoformat()
    )

    above20 = (
        price > sma20
        if sma20 is not None
        else False
    )

    above50 = (
        price > sma50
        if sma50 is not None
        else False
    )

    above200 = (
        price > sma200
        if sma200 is not None
        else False
    )

    return {
        "ticker":
            ticker,

        "price":
            round(price, 4),

        "price_date":
            price_date,

        "sma20":
            sma20,

        "sma50":
            sma50,

        "sma200":
            sma200,

        "sma50_slope_10d_pct":
            finite_or_none(
                slope,
                3,
            ),

        "atr14":
            finite_or_none(
                atr,
                4,
            ),

        "atr_ext_50":
            finite_or_none(
                atr_ext,
                3,
            ),

        "extension_zone":
            extension_zone(
                atr_ext
            ),

        "adr20_pct":
            finite_or_none(
                adr20(df),
                3,
            ),

        "ret_1m_pct":
            finite_or_none(
                close_return(
                    df,
                    21,
                ),
                3,
            ),

        "ret_3m_pct":
            finite_or_none(
                close_return(
                    df,
                    63,
                ),
                3,
            ),

        "ret_6m_pct":
            finite_or_none(
                close_return(
                    df,
                    126,
                ),
                3,
            ),

        "rs_spy_20d_pct":
            finite_or_none(
                relative_strength(
                    df,
                    spy_df,
                    20,
                ),
                3,
            ),

        "rs_spy_63d_pct":
            finite_or_none(
                relative_strength(
                    df,
                    spy_df,
                    63,
                ),
                3,
            ),

        "rs_qqq_63d_pct":
            finite_or_none(
                relative_strength(
                    df,
                    qqq_df,
                    63,
                ),
                3,
            ),

        "distance_52w_high_pct":
            finite_or_none(
                distance_52w_high(df),
                3,
            ),

        "range_10d_pct":
            finite_or_none(
                range_10_pct(df),
                3,
            ),

        "avg_dollar_volume_20d":
            finite_or_none(
                avg_dollar_volume(df),
                0,
            ),

        "volume_dry_up_ratio":
            finite_or_none(
                volume_dry_up_ratio(
                    df
                ),
                3,
            ),

        "trend_state":
            trend_state(
                price,
                sma20,
                sma50,
                sma200,
                slope,
            ),

        "above_20ma":
            bool(above20),

        "above_50ma":
            bool(above50),

        "above_200ma":
            bool(above200),
    }


def breadth_from_rows(rows):
    n = len(rows)

    if not n:
        return {
            "valid_tickers": 0,
            "pct_above_20ma": None,
            "pct_above_50ma": None,
            "pct_above_200ma": None,
            "strong_uptrend_pct": None,
            "rs_leader_pct": None,
            "extreme_overbought_count": 0,
            "extreme_oversold_count": 0,
        }

    strong = sum(
        1
        for row in rows
        if (
            row["trend_state"]
            == "STRONG_UPTREND"
        )
    )

    rs_leaders = sum(
        1
        for row in rows
        if (
            row.get(
                "rs_spy_63d_pct"
            )
            is not None
            and row[
                "rs_spy_63d_pct"
            ] > 10
        )
    )

    overbought = sum(
        1
        for row in rows
        if (
            row.get(
                "extension_zone"
            )
            == "EXTREME_OVERBOUGHT"
        )
    )

    oversold = sum(
        1
        for row in rows
        if (
            row.get(
                "extension_zone"
            )
            == "EXTREME_OVERSOLD"
        )
    )

    return {
        "valid_tickers":
            n,

        "pct_above_20ma":
            round(
                sum(
                    row[
                        "above_20ma"
                    ]
                    for row in rows
                )
                / n
                * 100.0,
                2,
            ),

        "pct_above_50ma":
            round(
                sum(
                    row[
                        "above_50ma"
                    ]
                    for row in rows
                )
                / n
                * 100.0,
                2,
            ),

        "pct_above_200ma":
            round(
                sum(
                    row[
                        "above_200ma"
                    ]
                    for row in rows
                )
                / n
                * 100.0,
                2,
            ),

        "strong_uptrend_pct":
            round(
                strong
                / n
                * 100.0,
                2,
            ),

        "rs_leader_pct":
            round(
                rs_leaders
                / n
                * 100.0,
                2,
            ),

        "extreme_overbought_count":
            overbought,

        "extreme_oversold_count":
            oversold,
    }


# ============================================================
# Intraday universe
# ============================================================

def build_intraday_universe_row(
    ticker,
    daily_df,
    intraday_df,
    spy_daily,
    qqq_daily,
    market_elapsed_minutes,
):
    daily_df = completed_daily(
        daily_df
    )

    if len(daily_df) < 220:
        return None

    if (
        intraday_df is None
        or intraday_df.empty
    ):
        return None

    intraday_df = (
        intraday_df
        .dropna(
            subset=[
                "Close",
                "High",
                "Low",
                "Volume",
            ]
        )
        .copy()
    )

    if intraday_df.empty:
        return None

    prev_close = float(
        daily_df["Close"]
        .iloc[-1]
    )

    prev_high = float(
        daily_df["High"]
        .iloc[-1]
    )

    avg_volume_20 = float(
        daily_df["Volume"]
        .iloc[-20:]
        .mean()
    )

    price = float(
        intraday_df[
            "Close"
        ].iloc[-1]
    )

    current_volume = float(
        intraday_df[
            "Volume"
        ].sum()
    )

    typical_price = (
        intraday_df["High"]
        + intraday_df["Low"]
        + intraday_df["Close"]
    ) / 3.0

    cumulative_volume = (
        intraday_df[
            "Volume"
        ]
        .cumsum()
    )

    cumulative_pv = (
        typical_price
        * intraday_df[
            "Volume"
        ]
    ).cumsum()

    vwap_series = (
        cumulative_pv
        / cumulative_volume
        .replace(0, float("nan"))
    )

    ema9_series = (
        intraday_df["Close"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    ema20_series = (
        intraday_df["Close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    vwap = finite_or_none(
        vwap_series.iloc[-1],
        4,
    )

    ema9 = finite_or_none(
        ema9_series.iloc[-1],
        4,
    )

    ema20 = finite_or_none(
        ema20_series.iloc[-1],
        4,
    )

    opening_bars = min(
        6,
        len(intraday_df),
    )

    opening_range_high = float(
        intraday_df["High"]
        .iloc[:opening_bars]
        .max()
    )

    day_change = pct_change(
        price,
        prev_close,
    )

    raw_rvol = None
    adjusted_rvol = None

    if avg_volume_20 > 0:
        raw_rvol = (
            current_volume
            / avg_volume_20
        )

        expected_fraction = max(
            min(
                market_elapsed_minutes
                / MARKET_SESSION_MINUTES,
                1.0,
            ),
            0.02,
        )

        adjusted_rvol = (
            raw_rvol
            / expected_fraction
        )

    turnover = (
        price
        * current_volume
    )

    distance_to_prev_high = None

    if prev_high > 0:
        distance_to_prev_high = (
            (
                price
                - prev_high
            )
            / prev_high
            * 100.0
        )

    above_vwap = (
        vwap is not None
        and price > vwap
    )

    above_ema9 = (
        ema9 is not None
        and price > ema9
    )

    ema9_above_ema20 = (
        ema9 is not None
        and ema20 is not None
        and ema9 > ema20
    )

    breaks_prev_high = (
        price > prev_high
    )

    breaks_opening_range = (
        price
        > opening_range_high
    )

    near_breakout = (
        not breaks_prev_high
        and distance_to_prev_high
        is not None
        and (
            -NEAR_BREAKOUT_DISTANCE_PCT
            <= distance_to_prev_high
            < 0
        )
        and day_change is not None
        and day_change > 0
        and above_vwap
    )

    high_rvol = (
        adjusted_rvol is not None
        and adjusted_rvol
        >= HIGH_RVOL_THRESHOLD
    )

    # ----------------------------
    # EOD context
    # ----------------------------

    close = (
        daily_df["Close"]
        .astype(float)
    )

    sma50 = float(
        close
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    atr = atr14(
        daily_df
    )

    atr_ext = None

    if (
        atr is not None
        and atr > 0
    ):
        atr_ext = (
            price - sma50
        ) / atr

    return {
        "ticker":
            ticker,

        "price":
            finite_or_none(
                price,
                4,
            ),

        "data_timestamp":
            str(
                intraday_df
                .index[-1]
            ),

        "prev_close":
            finite_or_none(
                prev_close,
                4,
            ),

        "prev_high":
            finite_or_none(
                prev_high,
                4,
            ),

        "day_change_pct":
            finite_or_none(
                day_change,
                3,
            ),

        "vwap":
            vwap,

        "ema9":
            ema9,

        "ema20":
            ema20,

        "opening_range_high":
            finite_or_none(
                opening_range_high,
                4,
            ),

        "current_volume":
            finite_or_none(
                current_volume,
                0,
            ),

        "raw_rvol":
            finite_or_none(
                raw_rvol,
                3,
            ),

        "adjusted_rvol":
            finite_or_none(
                adjusted_rvol,
                3,
            ),

        "turnover_m":
            finite_or_none(
                turnover
                / 1_000_000,
                2,
            ),

        "adr20_pct":
            finite_or_none(
                adr20(
                    daily_df
                ),
                3,
            ),

        "distance_to_prev_high_pct":
            finite_or_none(
                distance_to_prev_high,
                3,
            ),

        "above_vwap":
            bool(
                above_vwap
            ),

        "above_ema9":
            bool(
                above_ema9
            ),

        "ema9_above_ema20":
            bool(
                ema9_above_ema20
            ),

        "breaks_prev_high":
            bool(
                breaks_prev_high
            ),

        "breaks_opening_range":
            bool(
                breaks_opening_range
            ),

        "near_breakout":
            bool(
                near_breakout
            ),

        "high_rvol":
            bool(
                high_rvol
            ),

        # Daily context for ChatGPT
        "atr14":
            finite_or_none(
                atr,
                4,
            ),

        "atr_ext_50":
            finite_or_none(
                atr_ext,
                3,
            ),

        "extension_zone":
            extension_zone(
                atr_ext
            ),

        "ret_1m_pct":
            finite_or_none(
                close_return(
                    daily_df,
                    21,
                ),
                3,
            ),

        "ret_3m_pct":
            finite_or_none(
                close_return(
                    daily_df,
                    63,
                ),
                3,
            ),

        "rs_spy_63d_pct":
            finite_or_none(
                relative_strength(
                    daily_df,
                    spy_daily,
                    63,
                ),
                3,
            ),

        "rs_qqq_63d_pct":
            finite_or_none(
                relative_strength(
                    daily_df,
                    qqq_daily,
                    63,
                ),
                3,
            ),

        "distance_52w_high_pct":
            finite_or_none(
                distance_52w_high(
                    daily_df
                ),
                3,
            ),
    }


def build_intraday_breadth(
    rows,
):
    n = len(rows)

    if not n:
        return {
            "valid_tickers": 0,
            "positive_day_pct": None,
            "above_vwap_pct": None,
            "ema9_above_ema20_pct": None,
            "breakout_count": 0,
            "opening_range_breakout_count": 0,
            "near_breakout_count": 0,
            "high_rvol_count": 0,
        }

    positive = sum(
        1
        for row in rows
        if (
            row.get(
                "day_change_pct"
            )
            is not None
            and row[
                "day_change_pct"
            ] > 0
        )
    )

    above_vwap = sum(
        1
        for row in rows
        if row[
            "above_vwap"
        ]
    )

    ema_positive = sum(
        1
        for row in rows
        if row[
            "ema9_above_ema20"
        ]
    )

    breakout_count = sum(
        1
        for row in rows
        if row[
            "breaks_prev_high"
        ]
    )

    opening_breakouts = sum(
        1
        for row in rows
        if row[
            "breaks_opening_range"
        ]
    )

    near_breakout_count = sum(
        1
        for row in rows
        if row[
            "near_breakout"
        ]
    )

    high_rvol_count = sum(
        1
        for row in rows
        if row[
            "high_rvol"
        ]
    )

    return {
        "valid_tickers":
            n,

        "positive_day_pct":
            round(
                positive
                / n
                * 100.0,
                2,
            ),

        "above_vwap_pct":
            round(
                above_vwap
                / n
                * 100.0,
                2,
            ),

        "ema9_above_ema20_pct":
            round(
                ema_positive
                / n
                * 100.0,
                2,
            ),

        "breakout_count":
            breakout_count,

        "opening_range_breakout_count":
            opening_breakouts,

        "near_breakout_count":
            near_breakout_count,

        "high_rvol_count":
            high_rvol_count,
    }


# ============================================================
# Market / benchmark helpers
# ============================================================

def latest_intraday_snapshot(df):
    if (
        df is None
        or df.empty
        or "Close"
        not in df.columns
    ):
        return None

    clean = (
        df
        .dropna(
            subset=["Close"]
        )
        .copy()
    )

    if clean.empty:
        return None

    return {
        "price":
            float(
                clean["Close"]
                .iloc[-1]
            ),

        "data_timestamp":
            str(
                clean.index[-1]
            ),
    }


def moving_average_flags(df):
    df = completed_daily(df)

    if len(df) < 200:
        return {
            "above_20ma": None,
            "above_50ma": None,
            "above_200ma": None,
        }

    close = (
        df["Close"]
        .astype(float)
    )

    price = float(
        close.iloc[-1]
    )

    return {
        "above_20ma":
            bool(
                price
                > close
                .rolling(20)
                .mean()
                .iloc[-1]
            ),

        "above_50ma":
            bool(
                price
                > close
                .rolling(50)
                .mean()
                .iloc[-1]
            ),

        "above_200ma":
            bool(
                price
                > close
                .rolling(200)
                .mean()
                .iloc[-1]
            ),
    }


def benchmark_performance(df):
    df = completed_daily(df)

    if len(df) < 64:
        return None

    current = float(
        df["Close"]
        .iloc[-1]
    )

    previous = float(
        df["Close"]
        .iloc[-2]
    )

    return {
        "last_price":
            round(
                current,
                4,
            ),

        "previous_close":
            round(
                previous,
                4,
            ),

        "change_1d_pct":
            finite_or_none(
                pct_change(
                    current,
                    previous,
                ),
                3,
            ),

        "return_5d_pct":
            finite_or_none(
                close_return(
                    df,
                    5,
                ),
                3,
            ),

        "return_1m_pct":
            finite_or_none(
                close_return(
                    df,
                    21,
                ),
                3,
            ),

        "return_63d_pct":
            finite_or_none(
                close_return(
                    df,
                    63,
                ),
                3,
            ),

        "price_date":
            pd.Timestamp(
                df.index[-1]
            )
            .date()
            .isoformat(),
    }


def intraday_benchmark_performance(
    daily_df,
    intraday_df,
):
    base = benchmark_performance(
        daily_df
    )

    snap = latest_intraday_snapshot(
        intraday_df
    )

    if (
        base is None
        or snap is None
    ):
        return None

    previous = base[
        "last_price"
    ]

    current = snap[
        "price"
    ]

    base[
        "previous_close"
    ] = previous

    base[
        "last_price"
    ] = round(
        current,
        4,
    )

    base[
        "change_1d_pct"
    ] = finite_or_none(
        pct_change(
            current,
            previous,
        ),
        3,
    )

    base[
        "data_timestamp"
    ] = snap[
        "data_timestamp"
    ]

    return base


def sector_rankings(
    daily_map,
    mode,
    intraday_map=None,
):
    intraday_map = (
        intraday_map
        or {}
    )

    rows = []

    for (
        sector_name,
        ticker,
    ) in SECTOR_ETFS.items():

        daily = daily_map.get(
            ticker
        )

        if daily is None:
            continue

        if mode == "intraday":
            perf = (
                intraday_benchmark_performance(
                    daily,
                    intraday_map.get(
                        ticker
                    ),
                )
            )

        else:
            perf = (
                benchmark_performance(
                    daily
                )
            )

        if (
            perf is not None
            and perf.get(
                "change_1d_pct"
            )
            is not None
        ):
            rows.append(
                {
                    "sector":
                        sector_name,

                    "ticker":
                        ticker,

                    "change_1d_pct":
                        perf[
                            "change_1d_pct"
                        ],
                }
            )

    rows.sort(
        key=lambda row:
            row[
                "change_1d_pct"
            ],
        reverse=True,
    )

    return {
        "top_3":
            rows[:3],

        "bottom_3":
            rows[-3:][::-1],

        "all":
            rows,
    }


def classify_regime(
    spy_flags,
    qqq_flags,
    sox_flags,
    breadth,
    vix_level,
):
    score = 0
    reasons = []

    for (
        label,
        flags,
    ) in (
        ("SPY", spy_flags),
        ("QQQ", qqq_flags),
        ("SOX", sox_flags),
    ):
        if (
            flags.get(
                "above_20ma"
            )
            and flags.get(
                "above_50ma"
            )
        ):
            score += 1

            reasons.append(
                f"{label} above 20/50MA"
            )

        elif (
            flags.get(
                "above_20ma"
            )
            is False
            and flags.get(
                "above_50ma"
            )
            is False
        ):
            score -= 1

            reasons.append(
                f"{label} below 20/50MA"
            )

    if vix_level is not None:
        if vix_level < 20:
            score += 1

            reasons.append(
                "VIX below 20"
            )

        elif vix_level >= 25:
            score -= 1

            reasons.append(
                "VIX at/above 25"
            )

    positive_pct = breadth.get(
        "positive_day_pct"
    )

    if positive_pct is not None:
        if positive_pct >= 60:
            score += 1

            reasons.append(
                "broad participation positive"
            )

        elif positive_pct <= 40:
            score -= 1

            reasons.append(
                "broad participation weak"
            )

    above_vwap_pct = breadth.get(
        "above_vwap_pct"
    )

    if above_vwap_pct is not None:
        if above_vwap_pct >= 60:
            score += 1

            reasons.append(
                "majority above VWAP"
            )

        elif above_vwap_pct <= 40:
            score -= 1

            reasons.append(
                "majority below VWAP"
            )

    if score >= 3:
        state = "Risk-on"

    elif score <= -3:
        state = "Risk-off"

    else:
        state = "Neutral"

    confidence = min(
        100,
        50 + abs(score) * 10,
    )

    return (
        state,
        confidence,
        reasons,
    )


# ============================================================
# Mode
# ============================================================

def determine_mode(
    now_ny,
    forced,
):
    if forced in {
        "intraday",
        "eod",
    }:
        return forced

    if now_ny.weekday() >= 5:
        return "skip"

    minutes = (
        now_ny.hour * 60
        + now_ny.minute
    )

    # AUTO intraday intelligence only runs
    # during the first 2 regular-market hours.
    if (
        9 * 60 + 30
        <= minutes
        <= 11 * 60 + 30
    ):
        return "intraday"

    # Give yfinance time to publish
    # completed EOD candle.
    if (
        16 * 60 + 20
        <= minutes
        <= 18 * 60
    ):
        return "eod"

    return "skip"


# ============================================================
# Atomic JSON write
# ============================================================

def atomic_json_write(
    path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        f.write("\n")

    os.replace(
        temp,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "auto",
            "intraday",
            "eod",
        ],
        default="auto",
    )

    args = parser.parse_args()

    generated = (
        datetime.now(UTC)
        .replace(
            microsecond=0
        )
    )

    now_ny = (
        generated
        .astimezone(NY)
    )

    mode = determine_mode(
        now_ny,
        args.mode,
    )

    if mode == "skip":
        print(
            "Outside market "
            "intelligence window. "
            f"NY={now_ny.isoformat()}"
        )

        return 0

    universe = load_universe()

    print("=" * 80)

    print(
        f"Mode: {mode}"
    )

    print(
        f"NY: "
        f"{now_ny.isoformat()}"
    )

    print(
        f"Universe: "
        f"{len(universe)}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Daily universe
    # --------------------------------------------------------

    (
        daily_map,
        daily_failures,
    ) = download_batched(
        universe,
        "18mo",
        "1d",
        batch_size=DAILY_BATCH_SIZE,
    )

    daily_coverage = (
        len(daily_map)
        / len(universe)
    )

    if (
        daily_coverage
        < MIN_DAILY_COVERAGE
    ):
        raise RuntimeError(
            "Daily universe coverage "
            f"too low: "
            f"{len(daily_map)}/"
            f"{len(universe)}"
        )

    # --------------------------------------------------------
    # Benchmarks
    # --------------------------------------------------------

    benchmark_symbols = list(
        dict.fromkeys(
            [
                "SPY",
                "QQQ",
                "^SOX",
                "SOXX",
                "^VIX",
                *SECTOR_ETFS.values(),
            ]
        )
    )

    (
        bench_daily,
        benchmark_failures,
    ) = download_batched(
        benchmark_symbols,
        "18mo",
        "1d",
        batch_size=len(
            benchmark_symbols
        ),
    )

    for ticker in (
        "SPY",
        "QQQ",
        "SOXX",
        "^VIX",
    ):
        if ticker not in bench_daily:
            raise RuntimeError(
                "Missing critical "
                f"benchmark {ticker}"
            )

    sox_symbol = (
        "^SOX"
        if (
            "^SOX"
            in bench_daily
            and len(
                completed_daily(
                    bench_daily[
                        "^SOX"
                    ]
                )
            ) >= 64
        )
        else "SOXX"
    )

    # --------------------------------------------------------
    # Intraday mode:
    # remove today's partial daily bar
    # --------------------------------------------------------

    if mode == "intraday":
        session_date = (
            now_ny.date()
        )

        daily_map = {
            ticker:
                strip_current_session_daily_bar(
                    df,
                    session_date,
                )

            for (
                ticker,
                df,
            ) in daily_map.items()
        }

        bench_daily = {
            ticker:
                strip_current_session_daily_bar(
                    df,
                    session_date,
                )

            for (
                ticker,
                df,
            ) in bench_daily.items()
        }

    # --------------------------------------------------------
    # Intraday downloads
    # --------------------------------------------------------

    intraday_map = {}
    bench_intraday = {}

    if mode == "intraday":

        (
            intraday_map,
            intraday_failures,
        ) = download_batched(
            universe,
            "1d",
            "5m",
            batch_size=INTRADAY_BATCH_SIZE,
        )

        intraday_coverage = (
            len(intraday_map)
            / len(universe)
        )

        if (
            intraday_coverage
            < MIN_INTRADAY_COVERAGE
        ):
            raise RuntimeError(
                "Intraday universe "
                "coverage too low: "
                f"{len(intraday_map)}/"
                f"{len(universe)}"
            )

        active_benchmarks = [
            "SPY",
            "QQQ",
            sox_symbol,
            "^VIX",
            *SECTOR_ETFS.values(),
        ]

        (
            bench_intraday,
            _,
        ) = download_batched(
            active_benchmarks,
            "1d",
            "5m",
            batch_size=len(
                active_benchmarks
            ),
        )

    # ========================================================
    # EOD
    # ========================================================

    if mode == "eod":

        rows = []

        for ticker in universe:

            df = daily_map.get(
                ticker
            )

            if df is None:
                continue

            row = build_universe_row(
                ticker,
                df,
                bench_daily["SPY"],
                bench_daily["QQQ"],
            )

            if row is not None:
                rows.append(row)

        coverage = (
            len(rows)
            / len(universe)
        )

        if coverage < 0.80:
            raise RuntimeError(
                "EOD metric coverage "
                f"too low: "
                f"{len(rows)}/"
                f"{len(universe)}"
            )

        rows.sort(
            key=lambda row:
                row["ticker"]
        )

        dates = [
            row["price_date"]
            for row in rows
            if row.get(
                "price_date"
            )
        ]

        market_date = (
            max(dates)
            if dates
            else now_ny
            .date()
            .isoformat()
        )

        universe_payload = {
            "schema_version": 2,

            "generated_at_utc":
                generated
                .isoformat()
                .replace(
                    "+00:00",
                    "Z",
                ),

            "generated_at_hk":
                generated
                .astimezone(HK)
                .isoformat(),

            "mode":
                "EOD_UNIVERSE",

            "market_date":
                market_date,

            "universe_size":
                len(universe),

            "valid_ticker_count":
                len(rows),

            "failed_ticker_count":
                len(
                    set(
                        daily_failures
                    )
                ),

            "failures":
                sorted(
                    set(
                        daily_failures
                    )
                ),

            "breadth":
                breadth_from_rows(
                    rows
                ),

            "definitions": {
                "atr_ext_50":
                    "(close - SMA50) / ATR14",

                "extension_zone":
                    "Extreme oversold <= -5 ATR; "
                    "depressed <= -3; "
                    "extreme overbought >= +5 ATR",

                "sma50_slope_10d_pct":
                    "Percent change in SMA50 "
                    "versus 10 sessions earlier",

                "volume_dry_up_ratio":
                    "Average volume last 5 sessions / "
                    "average preceding 20 sessions",

                "rs":
                    "Stock return minus benchmark "
                    "return over named window",
            },

            "stocks":
                rows,
        }

        atomic_json_write(
            UNIVERSE_PATH,
            universe_payload,
        )

        print(
            f"Wrote "
            f"{UNIVERSE_PATH}"
        )

        # Market summary
        eod_breadth = {
            "positive_day_pct": None,
            "above_vwap_pct": None,
        }

    # ========================================================
    # INTRADAY
    # ========================================================

    else:

        open_minutes = (
            9 * 60 + 30
        )

        now_minutes = (
            now_ny.hour * 60
            + now_ny.minute
        )

        market_elapsed = max(
            1,
            min(
                now_minutes
                - open_minutes,
                MARKET_SESSION_MINUTES,
            ),
        )

        intraday_rows = []

        for ticker in universe:

            daily_df = (
                daily_map.get(
                    ticker
                )
            )

            intraday_df = (
                intraday_map.get(
                    ticker
                )
            )

            if (
                daily_df is None
                or intraday_df is None
            ):
                continue

            row = (
                build_intraday_universe_row(
                    ticker,
                    daily_df,
                    intraday_df,
                    bench_daily["SPY"],
                    bench_daily["QQQ"],
                    market_elapsed,
                )
            )

            if row is not None:
                intraday_rows.append(
                    row
                )

        coverage = (
            len(intraday_rows)
            / len(universe)
        )

        if (
            coverage
            < MIN_INTRADAY_COVERAGE
        ):
            raise RuntimeError(
                "Intraday metric "
                "coverage too low: "
                f"{len(intraday_rows)}/"
                f"{len(universe)}"
            )

        intraday_rows.sort(
            key=lambda row:
                row["ticker"]
        )

        timestamps = [
            row["data_timestamp"]

            for row
            in intraday_rows

            if row.get(
                "data_timestamp"
            )
        ]

        intraday_breadth = (
            build_intraday_breadth(
                intraday_rows
            )
        )

        intraday_payload = {
            "schema_version": 2,

            "generated_at_utc":
                generated
                .isoformat()
                .replace(
                    "+00:00",
                    "Z",
                ),

            "generated_at_hk":
                generated
                .astimezone(HK)
                .isoformat(),

            "mode":
                "INTRADAY_UNIVERSE",

            "market_date":
                now_ny
                .date()
                .isoformat(),

            "data_timestamp":
                (
                    max(timestamps)
                    if timestamps
                    else None
                ),

            "universe_size":
                len(universe),

            "valid_ticker_count":
                len(
                    intraday_rows
                ),

            "failed_ticker_count":
                len(
                    universe
                )
                - len(
                    intraday_rows
                ),

            "breadth":
                intraday_breadth,

            "definitions": {
                "distance_to_prev_high_pct":
                    "(price - previous session high) "
                    "/ previous session high * 100",

                "adjusted_rvol":
                    "Cumulative volume / "
                    "20-session average volume / "
                    "fraction of regular session elapsed",

                "near_breakout":
                    "Price is within 1% below previous "
                    "session high, day is positive, "
                    "and price is above VWAP",

                "high_rvol":
                    "Adjusted RVOL >= 2",

                "atr_ext_50":
                    "(current intraday price - "
                    "completed-session SMA50) / ATR14",
            },

            "stocks":
                intraday_rows,
        }

        atomic_json_write(
            INTRADAY_UNIVERSE_PATH,
            intraday_payload,
        )

        print(
            "Wrote "
            f"{INTRADAY_UNIVERSE_PATH}"
        )

        eod_breadth = (
            intraday_breadth
        )

    # ========================================================
    # latest_market.json
    # ========================================================

    spy_daily = (
        bench_daily["SPY"]
    )

    qqq_daily = (
        bench_daily["QQQ"]
    )

    sox_daily = (
        bench_daily[
            sox_symbol
        ]
    )

    vix_daily = (
        bench_daily["^VIX"]
    )

    if mode == "intraday":

        spy_perf = (
            intraday_benchmark_performance(
                spy_daily,
                bench_intraday.get(
                    "SPY"
                ),
            )
        )

        qqq_perf = (
            intraday_benchmark_performance(
                qqq_daily,
                bench_intraday.get(
                    "QQQ"
                ),
            )
        )

        sox_perf = (
            intraday_benchmark_performance(
                sox_daily,
                bench_intraday.get(
                    sox_symbol
                ),
            )
        )

        vix_perf = (
            intraday_benchmark_performance(
                vix_daily,
                bench_intraday.get(
                    "^VIX"
                ),
            )
        )

    else:

        spy_perf = (
            benchmark_performance(
                spy_daily
            )
        )

        qqq_perf = (
            benchmark_performance(
                qqq_daily
            )
        )

        sox_perf = (
            benchmark_performance(
                sox_daily
            )
        )

        vix_perf = (
            benchmark_performance(
                vix_daily
            )
        )

    if any(
        value is None
        for value in (
            spy_perf,
            qqq_perf,
            sox_perf,
            vix_perf,
        )
    ):
        raise RuntimeError(
            "Critical benchmark "
            "data incomplete"
        )

    spy_flags = (
        moving_average_flags(
            spy_daily
        )
    )

    qqq_flags = (
        moving_average_flags(
            qqq_daily
        )
    )

    sox_flags = (
        moving_average_flags(
            sox_daily
        )
    )

    regime_state, (
        regime_confidence
    ), regime_reasons = (
        classify_regime(
            spy_flags,
            qqq_flags,
            sox_flags,
            eod_breadth,
            vix_perf.get(
                "last_price"
            ),
        )
    )

    benchmark_dates = [
        value.get(
            "price_date"
        )
        for value in (
            spy_perf,
            qqq_perf,
            sox_perf,
            vix_perf,
        )
        if value.get(
            "price_date"
        )
    ]

    market_date = (
        now_ny
        .date()
        .isoformat()
        if mode == "intraday"
        else max(
            benchmark_dates
        )
    )

    market_payload = {
        "schema_version": 2,

        "generated_at_utc":
            generated
            .isoformat()
            .replace(
                "+00:00",
                "Z",
            ),

        "generated_at_hk":
            generated
            .astimezone(HK)
            .isoformat(),

        "market_date":
            market_date,

        "market_mode":
            mode,

        "mode":
            (
                "MARKET_INTRADAY"
                if mode
                == "intraday"
                else "MARKET_EOD"
            ),

        "market_regime": {
            "state":
                regime_state,

            "confidence":
                regime_confidence,

            "reasons":
                regime_reasons,
        },

        "benchmarks": {
            "SPY":
                spy_perf,

            "QQQ":
                qqq_perf,

            sox_symbol:
                sox_perf,

            "VIX":
                vix_perf,
        },

        "breadth":
            eod_breadth,

        "sector_performance":
            sector_rankings(
                bench_daily,
                mode,
                bench_intraday,
            ),

        "key_market_metrics": {
            "SPY":
                spy_flags,

            "QQQ":
                qqq_flags,

            sox_symbol:
                sox_flags,

            "VIX": {
                "level":
                    vix_perf.get(
                        "last_price"
                    ),

                "change_1d_pct":
                    vix_perf.get(
                        "change_1d_pct"
                    ),
            },
        },

        "coverage": {
            "configured_universe":
                len(universe),

            "daily_available":
                len(daily_map),

            "intraday_available":
                (
                    len(
                        intraday_map
                    )
                    if mode
                    == "intraday"
                    else None
                ),
        },
    }

    atomic_json_write(
        MARKET_PATH,
        market_payload,
    )

    print(
        f"Wrote "
        f"{MARKET_PATH}"
    )

    print("=" * 80)
    print("DONE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print(
            "MARKET_INTELLIGENCE_FAILED: "
            f"{exc}",
            file=sys.stderr,
        )

        raise
