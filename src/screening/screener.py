"""
Stock screening module for Qullamaggie Momentum / Breakout Setup.

Purpose:
- Find strong momentum stocks
- Avoid slow defensive stocks
- Prefer high ADR names
- Detect consolidation near highs
- Estimate pivot, stop, and risk %

This version is designed as a WATCHLIST generator, not an auto-buy system.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.storage import StockDatabase
from .indicators import calculate_rsi

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MIN_PRICE = 5.0
MIN_DATA_DAYS = 220

# Qullamaggie-style liquidity / volatility filters
MIN_ADR_20 = 4.0                  # Average Daily Range % over 20 days
MIN_AVG_DOLLAR_VOLUME = 20_000_000  # $20M average daily dollar volume

# Momentum filters
MAX_DISTANCE_FROM_52W_HIGH = 25.0
MIN_1M_RETURN = 10.0
MIN_3M_RETURN = 25.0

# Risk filter
MAX_SETUP_RISK_PCT = 12.0

# Consolidation
MIN_BASE_DAYS = 5
MAX_BASE_DAYS = 25

# These tickers are known stale / problematic.
# INFO is included because IHS Markit was acquired and is no longer a normal active public ticker.
INVALID_TICKERS = {
    "INFO"
}


# ---------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------

def _clean_price_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize price dataframe.
    Requires: Open, High, Low, Close, Volume if available.
    """
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()

    required_cols = ["High", "Low", "Close"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    df = df.dropna(subset=["High", "Low", "Close"])
    df = df.sort_index()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    return df


def _safe_pct_change(current: float, previous: float) -> float:
    if previous is None or previous == 0 or np.isnan(previous):
        return 0.0
    return ((current - previous) / previous) * 100.0


def _rolling_sma(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    value = series.rolling(window=window).mean().iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def calculate_adr(price_df: pd.DataFrame, period: int = 20) -> float:
    """
    ADR = Average Daily Range %.
    Formula: average((High - Low) / Close * 100)
    """
    df = _clean_price_df(price_df)

    if len(df) < period:
        return 0.0

    recent = df.iloc[-period:]
    adr_series = ((recent["High"] - recent["Low"]) / recent["Close"]) * 100
    adr = adr_series.mean()

    if pd.isna(adr):
        return 0.0

    return float(adr)


def calculate_avg_dollar_volume(price_df: pd.DataFrame, period: int = 20) -> float:
    """
    Average Dollar Volume = average(Close * Volume)
    """
    df = _clean_price_df(price_df)

    if len(df) < period or "Volume" not in df.columns:
        return 0.0

    recent = df.iloc[-period:]

    if recent["Volume"].isna().all():
        return 0.0

    dollar_volume = recent["Close"] * recent["Volume"]
    avg_dollar_volume = dollar_volume.mean()

    if pd.isna(avg_dollar_volume):
        return 0.0

    return float(avg_dollar_volume)


def calculate_returns(price_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate 1M / 3M / 6M returns.
    Approx trading days:
    - 1M = 21 days
    - 3M = 63 days
    - 6M = 126 days
    """
    df = _clean_price_df(price_df)

    if df.empty:
        return {"ret_1m": 0.0, "ret_3m": 0.0, "ret_6m": 0.0}

    close = df["Close"]
    current = float(close.iloc[-1])

    def get_return(days: int) -> float:
        if len(close) <= days:
            return 0.0
        previous = float(close.iloc[-days])
        return _safe_pct_change(current, previous)

    return {
        "ret_1m": get_return(21),
        "ret_3m": get_return(63),
        "ret_6m": get_return(126),
    }


def calculate_distance_from_52w_high(price_df: pd.DataFrame) -> float:
    """
    Distance from 52-week high.
    Lower is better.
    """
    df = _clean_price_df(price_df)

    if len(df) < 100:
        return 999.0

    recent = df.iloc[-252:] if len(df) >= 252 else df
    current = float(df["Close"].iloc[-1])
    high_52w = float(recent["High"].max())

    if high_52w <= 0:
        return 999.0

    distance = ((high_52w - current) / high_52w) * 100.0
    return float(distance)


def calculate_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    period: int = 63
) -> float:
    """
    Relative strength vs benchmark over selected period.
    Positive means stock outperformed benchmark.
    """
    stock_df = _clean_price_df(stock_df)

    if benchmark_df is None:
        return 0.0

    benchmark_df = _clean_price_df(benchmark_df)

    if len(stock_df) <= period or len(benchmark_df) <= period:
        return 0.0

    stock_current = float(stock_df["Close"].iloc[-1])
    stock_prev = float(stock_df["Close"].iloc[-period])

    bench_current = float(benchmark_df["Close"].iloc[-1])
    bench_prev = float(benchmark_df["Close"].iloc[-period])

    stock_return = _safe_pct_change(stock_current, stock_prev)
    bench_return = _safe_pct_change(bench_current, bench_prev)

    return float(stock_return - bench_return)


def calculate_volume_dry_up(price_df: pd.DataFrame) -> float:
    """
    Volume dry-up score.
    Qullamaggie-style bases often have reduced volume during consolidation.

    Score:
    - Recent 5D volume < 60% of previous 20D average = 100
    - < 80% = 70
    - < 100% = 40
    - otherwise = 0
    """
    df = _clean_price_df(price_df)

    if len(df) < 30 or "Volume" not in df.columns:
        return 0.0

    volume = df["Volume"]

    if volume.isna().all():
        return 0.0

    recent_5 = volume.iloc[-5:].mean()
    previous_20 = volume.iloc[-25:-5].mean()

    if pd.isna(recent_5) or pd.isna(previous_20) or previous_20 <= 0:
        return 0.0

    ratio = recent_5 / previous_20

    if ratio <= 0.60:
        return 100.0
    elif ratio <= 0.80:
        return 70.0
    elif ratio <= 1.00:
        return 40.0
    else:
        return 0.0


def detect_volume_spike_simple(price_df: pd.DataFrame) -> bool:
    """
    Breakout-style volume spike.
    Today's volume > 1.5x previous 20-day average.
    """
    df = _clean_price_df(price_df)

    if len(df) < 25 or "Volume" not in df.columns:
        return False

    volume = df["Volume"]

    if volume.isna().all():
        return False

    current_volume = volume.iloc[-1]
    avg_volume = volume.iloc[-21:-1].mean()

    if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume <= 0:
        return False

    return bool(current_volume > avg_volume * 1.5)


def estimate_pivot_stop_risk(price_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Estimate pivot, stop and risk.

    Simple logic:
    - Pivot = highest high in last 10 trading days
    - Stop = lowest low in last 10 trading days
    - Risk % = (pivot - stop) / pivot * 100

    This is not perfect, but useful for Telegram watchlist.
    """
    df = _clean_price_df(price_df)

    if len(df) < 15:
        return {
            "pivot": None,
            "stop": None,
            "risk_pct": None,
        }

    recent = df.iloc[-10:]
    pivot = float(recent["High"].max())
    stop = float(recent["Low"].min())

    if pivot <= 0 or stop <= 0 or stop >= pivot:
        risk_pct = None
    else:
        risk_pct = ((pivot - stop) / pivot) * 100.0

    return {
        "pivot": round(pivot, 2),
        "stop": round(stop, 2),
        "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
    }


# ---------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------

def calculate_momentum_score(
    price_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Qullamaggie-style momentum score.

    This is stricter than the old version.
    It avoids giving too much score just because price is above moving averages.
    """
    df = _clean_price_df(price_df)

    if len(df) < 200:
        return 0.0, {}

    close = df["Close"]
    current_price = float(close.iloc[-1])

    sma_10 = _rolling_sma(close, 10)
    sma_20 = _rolling_sma(close, 20)
    sma_50 = _rolling_sma(close, 50)
    sma_200 = _rolling_sma(close, 200)

    returns = calculate_returns(df)
    distance_52w = calculate_distance_from_52w_high(df)

    rs_spy = calculate_relative_strength(df, spy_df, period=63)
    rs_qqq = calculate_relative_strength(df, qqq_df, period=63)

    score = 0.0

    # Trend score: max 25
    if sma_10 and current_price > sma_10:
        score += 5
    if sma_20 and current_price > sma_20:
        score += 5
    if sma_50 and current_price > sma_50:
        score += 7
    if sma_200 and current_price > sma_200:
        score += 8

    # Returns score: max 30
    if returns["ret_1m"] >= 20:
        score += 10
    elif returns["ret_1m"] >= 10:
        score += 5

    if returns["ret_3m"] >= 50:
        score += 12
    elif returns["ret_3m"] >= 25:
        score += 7

    if returns["ret_6m"] >= 100:
        score += 8
    elif returns["ret_6m"] >= 50:
        score += 4

    # Distance from 52W high: max 15
    if distance_52w <= 5:
        score += 15
    elif distance_52w <= 10:
        score += 10
    elif distance_52w <= 15:
        score += 5

    # Relative strength: max 20
    if rs_spy > 20:
        score += 10
    elif rs_spy > 10:
        score += 6
    elif rs_spy > 0:
        score += 3

    if rs_qqq > 20:
        score += 10
    elif rs_qqq > 10:
        score += 6
    elif rs_qqq > 0:
        score += 3

    # Current price above short MAs but not too extended: max 10
    if sma_10:
        extension_from_10sma = ((current_price - sma_10) / sma_10) * 100
        if 0 <= extension_from_10sma <= 8:
            score += 10
        elif 8 < extension_from_10sma <= 15:
            score += 4

    details = {
        "current_price": current_price,
        "sma_10": sma_10 or 0.0,
        "sma_20": sma_20 or 0.0,
        "sma_50": sma_50 or 0.0,
        "sma_200": sma_200 or 0.0,
        "ret_1m": returns["ret_1m"],
        "ret_3m": returns["ret_3m"],
        "ret_6m": returns["ret_6m"],
        "distance_52w": distance_52w,
        "rs_spy": rs_spy,
        "rs_qqq": rs_qqq,
    }

    return min(score, 100.0), details


def calculate_consolidation_score(price_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Detect high tight consolidation.

    Important:
    Low volatility alone is NOT enough.
    We also reward:
    - being near 52W high
    - previous impulse
    - volume dry-up
    - controlled risk
    """
    df = _clean_price_df(price_df)

    if len(df) < 80:
        return 0.0, {}

    current_price = float(df["Close"].iloc[-1])

    # Last 10-day range
    recent_10 = df.iloc[-10:]
    high_10 = float(recent_10["High"].max())
    low_10 = float(recent_10["Low"].min())

    if low_10 <= 0:
        return 0.0, {}

    range_10_pct = ((high_10 - low_10) / low_10) * 100.0

    # Previous impulse move: compare current price to price 50 trading days ago
    price_50_days_ago = float(df["Close"].iloc[-50])
    impulse_50d = _safe_pct_change(current_price, price_50_days_ago)

    distance_52w = calculate_distance_from_52w_high(df)
    dry_up_score = calculate_volume_dry_up(df)
    pivot_data = estimate_pivot_stop_risk(df)
    risk_pct = pivot_data.get("risk_pct")

    score = 0.0

    # Tightness: max 30
    if range_10_pct <= 5:
        score += 30
    elif range_10_pct <= 8:
        score += 22
    elif range_10_pct <= 12:
        score += 12

    # Prior impulse: max 20
    if impulse_50d >= 50:
        score += 20
    elif impulse_50d >= 30:
        score += 15
    elif impulse_50d >= 20:
        score += 8

    # Near 52W high: max 15
    if distance_52w <= 5:
        score += 15
    elif distance_52w <= 10:
        score += 10
    elif distance_52w <= 15:
        score += 5

    # Volume dry-up: max 20
    if dry_up_score >= 100:
        score += 20
    elif dry_up_score >= 70:
        score += 14
    elif dry_up_score >= 40:
        score += 7

    # Risk quality: max 15
    if risk_pct is not None:
        if risk_pct <= 6:
            score += 15
        elif risk_pct <= 8:
            score += 10
        elif risk_pct <= 12:
            score += 5

    details = {
        "range_10_pct": range_10_pct,
        "impulse_50d": impulse_50d,
        "distance_52w": distance_52w,
        "volume_dry_up_score": dry_up_score,
        "pivot": pivot_data.get("pivot") or 0.0,
        "stop": pivot_data.get("stop") or 0.0,
        "risk_pct": risk_pct if risk_pct is not None else 999.0,
    }

    return min(score, 100.0), details


def build_reasons(
    ticker: str,
    metrics: Dict[str, float],
    hard_rejects: List[str]
) -> Tuple[str, str]:
    """
    Human-readable reason_in and reason_out for Telegram.
    """
    reason_in = []
    reason_out = []

    if metrics.get("adr_20", 0) >= MIN_ADR_20:
        reason_in.append(f"ADR {metrics['adr_20']:.1f}%")

    if metrics.get("avg_dollar_volume", 0) >= MIN_AVG_DOLLAR_VOLUME:
        reason_in.append(f"流動性足夠 ${metrics['avg_dollar_volume'] / 1_000_000:.1f}M")

    if metrics.get("ret_3m", 0) >= MIN_3M_RETURN:
        reason_in.append(f"3M +{metrics['ret_3m']:.1f}%")

    if metrics.get("distance_52w", 999) <= 15:
        reason_in.append(f"近52週高位 {metrics['distance_52w']:.1f}%")

    if metrics.get("range_10_pct", 999) <= 12:
        reason_in.append(f"10日窄幅 {metrics['range_10_pct']:.1f}%")

    if metrics.get("volume_dry_up_score", 0) >= 40:
        reason_in.append("整理期縮量")

    if metrics.get("risk_pct", 999) <= MAX_SETUP_RISK_PCT:
        reason_in.append(f"風險 {metrics['risk_pct']:.1f}%")

    if metrics.get("volume_spike", False):
        reason_in.append("成交量突破")

    for item in hard_rejects:
        reason_out.append(item)

    if not reason_in:
        reason_in.append("技術條件一般，需人手睇圖確認")

    if not reason_out:
        reason_out.append("未見硬性淘汰條件")

    return "；".join(reason_in), "；".join(reason_out)


# ---------------------------------------------------------
# Main Screener
# ---------------------------------------------------------

def screen_candidates(
    db: StockDatabase,
    tickers_list: List[str],
    value_weight: float = 0.6,
    support_weight: float = 0.4,
    min_data_days: int = MIN_DATA_DAYS
) -> pd.DataFrame:
    """
    Main screening function.

    Notes:
    - value_weight = momentum score weight
    - support_weight = setup / consolidation score weight

    Returns:
    DataFrame sorted by buy_signal.
    """

    if not tickers_list:
        logger.warning("Empty ticker list provided")
        return pd.DataFrame()

    logger.info(f"Screening {len(tickers_list)} candidates...")

    results = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=min_data_days + 180)

    # -----------------------------------------------------
    # Load benchmark data for RS calculation
    # -----------------------------------------------------

    spy_history = pd.DataFrame()
    qqq_history = pd.DataFrame()

    try:
        spy_history = db.get_price_history(
            "SPY",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        spy_history = _clean_price_df(spy_history)
    except Exception as e:
        logger.warning(f"Could not load SPY benchmark: {e}")

    try:
        qqq_history = db.get_price_history(
            "QQQ",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        qqq_history = _clean_price_df(qqq_history)
    except Exception as e:
        logger.warning(f"Could not load QQQ benchmark: {e}")

    # -----------------------------------------------------
    # Screening Loop
    # -----------------------------------------------------

    for ticker in tickers_list:
        try:
            ticker = ticker.upper().strip()

            if not ticker:
                continue

            hard_rejects = []

            if ticker in INVALID_TICKERS:
                logger.info(f"{ticker}: skipped because ticker is marked invalid/stale")
                continue

            logger.debug(f"Processing {ticker}...")

            fundamentals = db.get_latest_fundamentals(ticker)

            if not fundamentals:
                logger.warning(f"No fundamentals found for {ticker}")
                continue

            price_history = db.get_price_history(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            price_history = _clean_price_df(price_history)

            if price_history.empty or len(price_history) < min_data_days:
                logger.warning(f"Insufficient price data for {ticker}: {len(price_history)} days")
                continue

            current_price = fundamentals.get("current_price")

            if current_price is None or current_price <= 0:
                current_price = float(price_history["Close"].iloc[-1])

            if current_price is None or current_price <= 0:
                logger.warning(f"Invalid current price for {ticker}")
                continue

            current_price = float(current_price)

            # -------------------------------------------------
            # Core Metrics
            # -------------------------------------------------

            adr_20 = calculate_adr(price_history, period=20)
            avg_dollar_volume = calculate_avg_dollar_volume(price_history, period=20)
            returns = calculate_returns(price_history)
            distance_52w = calculate_distance_from_52w_high(price_history)

            momentum_score, momentum_details = calculate_momentum_score(
                price_history,
                spy_df=spy_history,
                qqq_df=qqq_history
            )

            consolidation_score, consolidation_details = calculate_consolidation_score(price_history)

            pivot_data = estimate_pivot_stop_risk(price_history)
            volume_spike = detect_volume_spike_simple(price_history)

            rsi = None
            if len(price_history) >= 15:
                rsi_series = calculate_rsi(price_history["Close"], period=14)
                if not rsi_series.isna().all():
                    rsi = float(rsi_series.iloc[-1])

            sma_10 = _rolling_sma(price_history["Close"], 10)
            sma_20 = _rolling_sma(price_history["Close"], 20)
            sma_50 = _rolling_sma(price_history["Close"], 50)
            sma_200 = _rolling_sma(price_history["Close"], 200)

            # -------------------------------------------------
            # Hard Rejects
            # -------------------------------------------------

            if current_price < MIN_PRICE:
                hard_rejects.append(f"價格低於 ${MIN_PRICE}")

            if adr_20 < MIN_ADR_20:
                hard_rejects.append(f"ADR太低 {adr_20:.1f}%")

            if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
                hard_rejects.append(f"成交額不足 ${avg_dollar_volume / 1_000_000:.1f}M")

            if distance_52w > MAX_DISTANCE_FROM_52W_HIGH:
                hard_rejects.append(f"離52週高位太遠 {distance_52w:.1f}%")

            if returns["ret_1m"] < MIN_1M_RETURN and returns["ret_3m"] < MIN_3M_RETURN:
                hard_rejects.append(
                    f"動能不足 1M {returns['ret_1m']:.1f}% / 3M {returns['ret_3m']:.1f}%"
                )

            risk_pct = pivot_data.get("risk_pct")

            if risk_pct is not None and risk_pct > MAX_SETUP_RISK_PCT:
                hard_rejects.append(f"setup風險太闊 {risk_pct:.1f}%")

            if sma_20 is not None and current_price < sma_20:
                hard_rejects.append("跌穿20日線")

            # -------------------------------------------------
            # Buy Signal Score
            # -------------------------------------------------

            buy_signal = (momentum_score * value_weight) + (consolidation_score * support_weight)

            # Bonus only if already passing basic volatility/liquidity conditions
            if volume_spike and adr_20 >= MIN_ADR_20 and consolidation_score >= 50:
                buy_signal += 5

            # Penalty for hard rejects
            # Important: not always immediate delete, but strongly demote.
            penalty = len(hard_rejects) * 18
            buy_signal -= penalty

            buy_signal = max(0.0, min(buy_signal, 100.0))

            # -------------------------------------------------
            # Metrics for output
            # -------------------------------------------------

            metrics = {
                "adr_20": adr_20,
                "avg_dollar_volume": avg_dollar_volume,
                "ret_1m": returns["ret_1m"],
                "ret_3m": returns["ret_3m"],
                "ret_6m": returns["ret_6m"],
                "distance_52w": distance_52w,
                "range_10_pct": consolidation_details.get("range_10_pct", 999.0),
                "impulse_50d": consolidation_details.get("impulse_50d", 0.0),
                "volume_dry_up_score": consolidation_details.get("volume_dry_up_score", 0.0),
                "risk_pct": risk_pct if risk_pct is not None else 999.0,
                "volume_spike": volume_spike,
            }

            reason_in, reason_out = build_reasons(ticker, metrics, hard_rejects)

            # -------------------------------------------------
            # Result Row
            # -------------------------------------------------

            results.append({
                "ticker": ticker,
                "name": fundamentals.get("name", ticker),
                "sector": fundamentals.get("sector", "Unknown"),

                "current_price": round(current_price, 2),

                "buy_signal": round(buy_signal, 2),
                "momentum_score": round(momentum_score, 2),
                "consolidation_score": round(consolidation_score, 2),

                "adr_20": round(adr_20, 2),
                "avg_dollar_volume_m": round(avg_dollar_volume / 1_000_000, 2),

                "ret_1m": round(returns["ret_1m"], 2),
                "ret_3m": round(returns["ret_3m"], 2),
                "ret_6m": round(returns["ret_6m"], 2),

                "distance_52w": round(distance_52w, 2),

                "rs_spy": round(momentum_details.get("rs_spy", 0.0), 2),
                "rs_qqq": round(momentum_details.get("rs_qqq", 0.0), 2),

                "range_10_pct": round(consolidation_details.get("range_10_pct", 999.0), 2),
                "impulse_50d": round(consolidation_details.get("impulse_50d", 0.0), 2),
                "volume_dry_up_score": round(consolidation_details.get("volume_dry_up_score", 0.0), 2),
                "volume_spike": volume_spike,

                "pivot": pivot_data.get("pivot"),
                "stop": pivot_data.get("stop"),
                "risk_pct": risk_pct,

                "rsi": round(rsi, 2) if rsi is not None else None,

                "sma_10": round(sma_10, 2) if sma_10 is not None else None,
                "sma_20": round(sma_20, 2) if sma_20 is not None else None,
                "sma_50": round(sma_50, 2) if sma_50 is not None else None,
                "sma_200": round(sma_200, 2) if sma_200 is not None else None,

                "pe_ratio": fundamentals.get("pe_ratio"),
                "pb_ratio": fundamentals.get("pb_ratio"),

                "reason_in": reason_in,
                "reason_out": reason_out,

                "hard_reject_count": len(hard_rejects),
                "data_points": len(price_history),
            })

            logger.info(
                f"{ticker}: Signal={buy_signal:.1f} "
                f"Momentum={momentum_score:.1f} "
                f"Setup={consolidation_score:.1f} "
                f"ADR={adr_20:.1f}% "
                f"Risk={risk_pct if risk_pct is not None else 'N/A'} "
                f"Rejects={len(hard_rejects)}"
            )

        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            continue

    if not results:
        logger.warning("No valid screening results")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Keep only meaningful candidates.
    # You can change 50 to 40 if too few results.
    df = df[df["buy_signal"] >= 50].copy()

    if df.empty:
        logger.warning("No candidates passed minimum buy_signal threshold")
        return pd.DataFrame()

    # Better ranking:
    # 1. Higher buy signal
    # 2. Fewer hard rejects
    # 3. Higher ADR
    # 4. Stronger 3M return
    df = df.sort_values(
        by=["buy_signal", "hard_reject_count", "adr_20", "ret_3m"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

    logger.info(f"Successfully screened {len(df)} candidates after filters")

    return df
