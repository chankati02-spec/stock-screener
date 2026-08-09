#!/usr/bin/env python3
"""Generate compact market intelligence JSON for ChatGPT analysis.

This module is deliberately isolated from the existing screener so it cannot
change signal thresholds, Telegram alerts, or the existing latest*.json files.
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

NY = ZoneInfo("America/New_York")
HK = ZoneInfo("Asia/Hong_Kong")
UTC = ZoneInfo("UTC")

WATCHLIST_DIR = Path("watchlists")
MARKET_PATH = WATCHLIST_DIR / "latest_market.json"
UNIVERSE_PATH = WATCHLIST_DIR / "latest_universe.json"

INVALID_TICKERS = {
    "INFO", "ATVI", "ALXN", "CELG", "CTXS", "MXIM", "NUAN", "ABC", "BLL",
    "COG", "DISCA", "DISCK", "DRE", "FISV", "KSU", "MYL", "NBL", "PEAK",
    "FB", "NDX", "ARKG", "INFQ", "SOLS", "IMSR", "SPCX", "EIP", "ILLM",
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


def clean_ticker(value):
    if value is None:
        return None
    ticker = str(value).upper().strip().split("#")[0].strip().replace(".", "-")
    if not ticker or ticker in INVALID_TICKERS:
        return None
    if not ticker.replace("-", "").isalnum():
        return None
    return ticker


def load_universe(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    out = []
    seen = set()

    for raw in cfg.get("stock_universe", []):
        ticker = clean_ticker(raw)
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)

    if not out:
        raise RuntimeError("config.yaml -> stock_universe is empty")

    return out


def flatten_single(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[-1]) == 1:
            df.columns = df.columns.get_level_values(0)

    return df


def extract_ticker_df(data, ticker, batch_len):
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        if batch_len == 1:
            return flatten_single(data)

        if not isinstance(data.columns, pd.MultiIndex):
            return pd.DataFrame()

        level0 = set(map(str, data.columns.get_level_values(0)))
        level_last = set(map(str, data.columns.get_level_values(-1)))

        if ticker in level0:
            return flatten_single(data[ticker].copy())

        if ticker in level_last:
            return flatten_single(data.xs(ticker, axis=1, level=-1).copy())

    except Exception:
        return pd.DataFrame()

    return pd.DataFrame()


def finite_or_none(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return round(value, digits)


def pct_change(current, previous):
    try:
        current = float(current)
        previous = float(previous)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(current) or not math.isfinite(previous) or previous == 0:
        return None

    return (current / previous - 1.0) * 100.0


def close_return(df, sessions):
    if df is None or df.empty or len(df) <= sessions:
        return None

    return pct_change(
        df["Close"].iloc[-1],
        df["Close"].iloc[-1 - sessions],
    )


def completed_daily(df):
    if df is None or df.empty:
        return pd.DataFrame()

    required = [c for c in ["Close", "High", "Low"] if c in df.columns]
    if len(required) < 3:
        return pd.DataFrame()

    return df.dropna(subset=["Close", "High", "Low"]).copy()


def strip_current_session_daily_bar(df, session_date):
    out = completed_daily(df)

    if out.empty:
        return out

    try:
        last_ts = pd.Timestamp(out.index[-1])

        if last_ts.tzinfo is not None:
            last_date = last_ts.tz_convert(NY).date()
        else:
            last_date = last_ts.date()

        if last_date >= session_date and len(out) >= 2:
            out = out.iloc[:-1].copy()

    except Exception:
        pass

    return out


def download_batched(tickers, period, interval, batch_size=45):
    result = {}
    failures = []

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]

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
            print(f"download failed for batch {start}: {exc}", file=sys.stderr)
            continue

        for ticker in batch:
            df = extract_ticker_df(raw, ticker, len(batch))

            if df is None or df.empty:
                failures.append(ticker)
            else:
                result[ticker] = df

    return result, failures


def latest_intraday_snapshot(df):
    if df is None or df.empty or "Close" not in df.columns:
        return None

    d = df.dropna(subset=["Close"]).copy()

    if d.empty:
        return None

    return {
        "price": float(d["Close"].iloc[-1]),
        "data_timestamp": str(d.index[-1]),
    }


def atr14(df):
    if df is None or len(df) < 15:
        return None

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev).abs(),
            (low - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    value = tr.rolling(14).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def range_10_pct(df):
    if df is None or len(df) < 10:
        return None

    recent = df.iloc[-10:]
    low = float(recent["Low"].min())
    high = float(recent["High"].max())

    if low <= 0:
        return None

    return (high - low) / low * 100.0


def distance_52w_high(df):
    if df is None or len(df) < 100:
        return None

    recent = df.iloc[-252:] if len(df) >= 252 else df
    high = float(recent["High"].max())
    close = float(df["Close"].iloc[-1])

    if high <= 0:
        return None

    return (high - close) / high * 100.0


def adr20(df):
    if df is None or len(df) < 20:
        return None

    recent = df.iloc[-20:]
    x = ((recent["High"] - recent["Low"]) / recent["Close"]) * 100.0

    return None if x.isna().all() else float(x.mean())


def avg_dollar_volume(df):
    if df is None or len(df) < 20 or "Volume" not in df.columns:
        return None

    recent = df.iloc[-20:]
    x = recent["Close"] * recent["Volume"]

    return None if x.isna().all() else float(x.mean())


def volume_dry_up_ratio(df):
    if df is None or len(df) < 25 or "Volume" not in df.columns:
        return None

    recent5 = float(df["Volume"].iloc[-5:].mean())
    prior20 = float(df["Volume"].iloc[-25:-5].mean())

    if not math.isfinite(prior20) or prior20 <= 0:
        return None

    return recent5 / prior20


def relative_strength(stock_df, bench_df, sessions):
    stock_ret = close_return(stock_df, sessions)
    bench_ret = close_return(bench_df, sessions)

    if stock_ret is None or bench_ret is None:
        return None

    return stock_ret - bench_ret


def extension_zone(value):
    if value is None:
        return "UNKNOWN"

    if value < 0:
        return "BELOW_50MA"
    if value < 1:
        return "NEAR_50MA"
    if value < 2:
        return "ABOVE_50MA"
    if value < 3:
        return "EXTENDED"

    return "VERY_EXTENDED"


def trend_state(price, sma20, sma50, sma200, slope):
    if any(v is None for v in (price, sma20, sma50, sma200)):
        return "UNKNOWN"

    if price > sma20 > sma50 > sma200 and slope is not None and slope > 0:
        return "STRONG_UPTREND"

    if price > sma50 > sma200 and slope is not None and slope > 0:
        return "UPTREND"

    if price > sma200 and price < sma50:
        return "PULLBACK_ABOVE_200MA"

    if price < sma50 and price < sma200:
        return "DOWNTREND"

    return "MIXED"


def build_universe_row(ticker, df, spy_df, qqq_df):
    df = completed_daily(df)

    if len(df) < 220:
        return None

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    sma20s = close.rolling(20).mean()
    sma50s = close.rolling(50).mean()
    sma200s = close.rolling(200).mean()

    sma20 = finite_or_none(sma20s.iloc[-1])
    sma50 = finite_or_none(sma50s.iloc[-1])
    sma200 = finite_or_none(sma200s.iloc[-1])

    slope = None

    if len(df) >= 60 and not pd.isna(sma50s.iloc[-11]) and sma50s.iloc[-11] != 0:
        slope = (
            float(sma50s.iloc[-1]) / float(sma50s.iloc[-11]) - 1.0
        ) * 100.0

    atr = atr14(df)

    ext = (
        (price - sma50) / atr
        if atr is not None and atr > 0 and sma50 is not None
        else None
    )

    price_date = pd.Timestamp(df.index[-1]).date().isoformat()

    above20 = price > sma20 if sma20 is not None else False
    above50 = price > sma50 if sma50 is not None else False
    above200 = price > sma200 if sma200 is not None else False

    return {
        "ticker": ticker,
        "price": round(price, 4),
        "price_date": price_date,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "sma50_slope_10d_pct": finite_or_none(slope, 3),
        "atr14": finite_or_none(atr, 4),
        "atr_ext_50": finite_or_none(ext, 3),
        "extension_zone": extension_zone(ext),
        "adr20_pct": finite_or_none(adr20(df), 3),
        "ret_1m_pct": finite_or_none(close_return(df, 21), 3),
        "ret_3m_pct": finite_or_none(close_return(df, 63), 3),
        "ret_6m_pct": finite_or_none(close_return(df, 126), 3),
        "rs_spy_20d_pct": finite_or_none(relative_strength(df, spy_df, 20), 3),
        "rs_spy_63d_pct": finite_or_none(relative_strength(df, spy_df, 63), 3),
        "rs_qqq_63d_pct": finite_or_none(relative_strength(df, qqq_df, 63), 3),
        "distance_52w_high_pct": finite_or_none(distance_52w_high(df), 3),
        "range_10d_pct": finite_or_none(range_10_pct(df), 3),
        "avg_dollar_volume_20d": finite_or_none(avg_dollar_volume(df), 0),
        "volume_dry_up_ratio": finite_or_none(volume_dry_up_ratio(df), 3),
        "trend_state": trend_state(price, sma20, sma50, sma200, slope),
        "above_20ma": bool(above20),
        "above_50ma": bool(above50),
        "above_200ma": bool(above200),
    }


def breadth_from_rows(rows):
    n = len(rows)

    if not n:
        return {
            "valid_tickers": 0,
            "pct_above_20ma": None,
            "pct_above_50ma": None,
            "pct_above_200ma": None,
        }

    return {
        "valid_tickers": n,
        "pct_above_20ma": round(
            sum(r["above_20ma"] for r in rows) / n * 100.0, 2
        ),
        "pct_above_50ma": round(
            sum(r["above_50ma"] for r in rows) / n * 100.0, 2
        ),
        "pct_above_200ma": round(
            sum(r["above_200ma"] for r in rows) / n * 100.0, 2
        ),
    }


def moving_average_flags(df):
    d = completed_daily(df)

    if len(d) < 200:
        return {
            "above_20ma": None,
            "above_50ma": None,
            "above_200ma": None,
        }

    close = d["Close"].astype(float)
    price = float(close.iloc[-1])

    return {
        "above_20ma": bool(price > close.rolling(20).mean().iloc[-1]),
        "above_50ma": bool(price > close.rolling(50).mean().iloc[-1]),
        "above_200ma": bool(price > close.rolling(200).mean().iloc[-1]),
    }


def benchmark_performance(df):
    d = completed_daily(df)

    if len(d) < 64:
        return None

    current = float(d["Close"].iloc[-1])
    previous = float(d["Close"].iloc[-2])

    return {
        "last_price": round(current, 4),
        "previous_close": round(previous, 4),
        "change_1d_pct": finite_or_none(pct_change(current, previous), 3),
        "return_5d_pct": finite_or_none(close_return(d, 5), 3),
        "return_1m_pct": finite_or_none(close_return(d, 21), 3),
        "return_20d_pct": finite_or_none(close_return(d, 20), 3),
        "return_63d_pct": finite_or_none(close_return(d, 63), 3),
        "price_date": pd.Timestamp(d.index[-1]).date().isoformat(),
    }


def intraday_benchmark_performance(daily_df, intraday_df):
    base = benchmark_performance(daily_df)
    snap = latest_intraday_snapshot(intraday_df)

    if base is None or snap is None:
        return None

    previous = base["last_price"]
    current = snap["price"]

    base["previous_close"] = previous
    base["last_price"] = round(current, 4)
    base["change_1d_pct"] = finite_or_none(pct_change(current, previous), 3)
    base["data_timestamp"] = snap["data_timestamp"]

    return base


def sector_rankings(daily_map, mode, intraday_map=None):
    rows = []
    intraday_map = intraday_map or {}

    for name, ticker in SECTOR_ETFS.items():
        df = daily_map.get(ticker)

        if df is None:
            continue

        perf = (
            intraday_benchmark_performance(
                df,
                intraday_map.get(ticker),
            )
            if mode == "intraday"
            else benchmark_performance(df)
        )

        if perf is not None and perf["change_1d_pct"] is not None:
            rows.append(
                {
                    "sector": name,
                    "ticker": ticker,
                    "change_1d_pct": perf["change_1d_pct"],
                }
            )

    rows.sort(key=lambda r: r["change_1d_pct"], reverse=True)

    return {
        "top_3": rows[:3],
        "bottom_3": list(reversed(rows[-3:])),
        "all": rows,
    }


def vix_five_day_trend(vix_df):
    value = close_return(completed_daily(vix_df), 5)

    if value is None:
        return "UNKNOWN"

    if value >= 5:
        return "RISING"

    if value <= -5:
        return "FALLING"

    return "FLAT"


def classify_regime(
    spy_flags,
    qqq_flags,
    sox_flags,
    vix_level,
    breadth20,
    adv,
    dec,
):
    score = 0
    reasons = []

    for label, flags in (
        ("SPY", spy_flags),
        ("QQQ", qqq_flags),
        ("SOX", sox_flags),
    ):
        if flags.get("above_20ma") and flags.get("above_50ma"):
            score += 1
            reasons.append(f"{label} above 20/50MA")
        elif (
            flags.get("above_20ma") is False
            and flags.get("above_50ma") is False
        ):
            score -= 1
            reasons.append(f"{label} below 20/50MA")

    if vix_level is not None:
        if vix_level < 20:
            score += 1
            reasons.append("VIX below 20")
        elif vix_level >= 25:
            score -= 1
            reasons.append("VIX at/above 25")

    if breadth20 is not None:
        if breadth20 >= 55:
            score += 1
            reasons.append("breadth above 20MA >=55%")
        elif breadth20 <= 45:
            score -= 1
            reasons.append("breadth above 20MA <=45%")

    if adv is not None and dec is not None and adv + dec > 0:
        participation = (adv - dec) / (adv + dec)

        if participation >= 0.15:
            score += 1
            reasons.append("advancers lead decliners")
        elif participation <= -0.15:
            score -= 1
            reasons.append("decliners lead advancers")

    regime = "Risk-on" if score >= 3 else "Risk-off" if score <= -3 else "Neutral"
    confidence = min(100, 50 + abs(score) * 10)

    return regime, confidence, reasons


def atomic_json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    with open(temp, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")

    os.replace(temp, path)


def determine_mode(now_ny, forced):
    if forced in {"intraday", "eod"}:
        return forced

    if now_ny.weekday() >= 5:
        return "skip"

    minute = now_ny.hour * 60 + now_ny.minute

    if 9 * 60 + 30 <= minute < 16 * 60:
        return "intraday"

    # Give yfinance some time to publish the completed daily candle.
    if now_ny.hour == 16 and 20 <= now_ny.minute <= 59:
        return "eod"

    return "skip"


def build_market(
    mode,
    universe,
    daily_map,
    intraday_map,
    bench_daily,
    bench_intraday,
    generated,
):
    spy = bench_daily["SPY"]
    qqq = bench_daily["QQQ"]

    sox_symbol = (
        "^SOX"
        if "^SOX" in bench_daily
        and len(completed_daily(bench_daily["^SOX"])) >= 64
        else "SOXX"
    )

    sox = bench_daily[sox_symbol]
    vix = bench_daily["^VIX"]

    if mode == "intraday":
        spy_perf = intraday_benchmark_performance(
            spy,
            bench_intraday.get("SPY"),
        )
        qqq_perf = intraday_benchmark_performance(
            qqq,
            bench_intraday.get("QQQ"),
        )
        sox_perf = intraday_benchmark_performance(
            sox,
            bench_intraday.get(sox_symbol),
        )
        vix_perf = intraday_benchmark_performance(
            vix,
            bench_intraday.get("^VIX"),
        )
    else:
        spy_perf = benchmark_performance(spy)
        qqq_perf = benchmark_performance(qqq)
        sox_perf = benchmark_performance(sox)
        vix_perf = benchmark_performance(vix)

    if any(x is None for x in (spy_perf, qqq_perf, sox_perf, vix_perf)):
        raise RuntimeError("critical benchmark fetch incomplete")

    adv = 0
    dec = 0
    unchanged = 0
    above20_count = 0
    above20_valid = 0
    available = 0

    for ticker in universe:
        d = completed_daily(daily_map.get(ticker))

        if len(d) < 21:
            continue

        if mode == "intraday":
            snap = latest_intraday_snapshot(intraday_map.get(ticker))

            if snap is None:
                continue

            price = snap["price"]
            previous = float(d["Close"].iloc[-1])
        else:
            if len(d) < 2:
                continue

            price = float(d["Close"].iloc[-1])
            previous = float(d["Close"].iloc[-2])

        sma20 = float(d["Close"].rolling(20).mean().iloc[-1])

        available += 1

        change = pct_change(price, previous)

        if change is not None:
            if change > 0.01:
                adv += 1
            elif change < -0.01:
                dec += 1
            else:
                unchanged += 1

        if math.isfinite(sma20):
            above20_valid += 1

            if price > sma20:
                above20_count += 1

    if available < max(1, int(len(universe) * 0.65)):
        raise RuntimeError(
            f"universe coverage too low: {available}/{len(universe)}; "
            "preserving last good JSON"
        )

    pct_above20 = (
        above20_count / above20_valid * 100.0
        if above20_valid
        else None
    )

    spy_flags = moving_average_flags(spy)
    qqq_flags = moving_average_flags(qqq)
    sox_flags = moving_average_flags(sox)

    regime, confidence, reasons = classify_regime(
        spy_flags,
        qqq_flags,
        sox_flags,
        vix_perf["last_price"],
        pct_above20,
        adv,
        dec,
    )

    timestamps = [
        x.get("data_timestamp")
        for x in (spy_perf, qqq_perf, sox_perf, vix_perf)
        if x.get("data_timestamp")
    ]

    data_timestamp = (
        max(timestamps)
        if timestamps
        else max(
            spy_perf["price_date"],
            qqq_perf["price_date"],
            sox_perf["price_date"],
            vix_perf["price_date"],
        )
    )

    # Use the latest completed benchmark trading session as market_date.
    # This avoids weekend/holiday manual runs being stamped with a non-trading date.
    market_date = max(
        spy_perf["price_date"],
        qqq_perf["price_date"],
        sox_perf["price_date"],
        vix_perf["price_date"],
    )

    return {
        "schema_version": 1,
        "date": market_date,
        "generated_at": generated.isoformat().replace("+00:00", "Z"),
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "generated_at_hk": generated.astimezone(HK).isoformat(),
        "data_timestamp": data_timestamp,
        "market_date": market_date,
        "market_mode": mode,
        "mode": (
            "MARKET_INTRADAY"
            if mode == "intraday"
            else "MARKET_EOD"
        ),
        "market_regime": {
            "state": regime,
            "confidence": confidence,
            "reasons": reasons,
        },
        "benchmarks": {
            "SPY": spy_perf,
            "QQQ": qqq_perf,
            sox_symbol: sox_perf,
            "VIX": vix_perf,
        },
        "breadth": {
            "advancing": adv,
            "declining": dec,
            "unchanged": unchanged,
            "pct_above_20ma": finite_or_none(pct_above20, 2),
            "coverage": available,
            "universe_size": len(universe),
        },
        "sector_performance": sector_rankings(
            bench_daily,
            mode,
            bench_intraday,
        ),
        "index_performance": {
            "SPY": {
                "1d_pct": spy_perf["change_1d_pct"],
                "5d_pct": spy_perf["return_5d_pct"],
                "1m_pct": spy_perf["return_1m_pct"],
            },
            "QQQ": {
                "1d_pct": qqq_perf["change_1d_pct"],
                "5d_pct": qqq_perf["return_5d_pct"],
                "1m_pct": qqq_perf["return_1m_pct"],
            },
            sox_symbol: {
                "1d_pct": sox_perf["change_1d_pct"],
                "5d_pct": sox_perf["return_5d_pct"],
                "1m_pct": sox_perf["return_1m_pct"],
            },
        },
        "key_market_metrics": {
            "SPY": spy_flags,
            "QQQ": qqq_flags,
            sox_symbol: sox_flags,
            "VIX": {
                "level": vix_perf["last_price"],
                "change_1d_pct": vix_perf["change_1d_pct"],
                "trend_5d": vix_five_day_trend(vix),
            },
        },
        "freshness": {
            "status": (
                "stale"
                if mode == "intraday"
                and timestamps
                and (
                    generated
                    - pd.Timestamp(max(timestamps)).to_pydatetime().astimezone(UTC)
                ).total_seconds() / 60.0 > 45
                else "ok"
            ),
            "benchmark_data_timestamp": data_timestamp,
            "age_minutes": (
                round(
                    max(
                        0.0,
                        (
                            generated
                            - pd.Timestamp(max(timestamps))
                            .to_pydatetime()
                            .astimezone(UTC)
                        ).total_seconds() / 60.0,
                    ),
                    1,
                )
                if timestamps
                else None
            ),
            "stale_after_minutes": 45 if mode == "intraday" else None,
            "universe_coverage_pct": round(
                available / len(universe) * 100.0,
                2,
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "intraday", "eod"],
        default="auto",
    )
    args = parser.parse_args()

    generated = datetime.now(UTC).replace(microsecond=0)
    now_ny = generated.astimezone(NY)
    mode = determine_mode(now_ny, args.mode)

    if mode == "skip":
        print(
            f"Outside US market intelligence window: "
            f"{now_ny.isoformat()}"
        )
        return 0

    universe = load_universe()

    print(
        f"mode={mode} universe={len(universe)} "
        f"generated={generated.isoformat()}"
    )

    daily_map, _ = download_batched(
        universe,
        "18mo",
        "1d",
        batch_size=45,
    )

    if len(daily_map) / len(universe) < 0.80:
        raise RuntimeError(
            f"daily universe coverage too low "
            f"({len(daily_map)}/{len(universe)}); "
            "last good JSON will be preserved"
        )

    benchmark_symbols = [
        "SPY",
        "QQQ",
        "^SOX",
        "SOXX",
        "^VIX",
        *SECTOR_ETFS.values(),
    ]

    bench_daily, _ = download_batched(
        benchmark_symbols,
        "18mo",
        "1d",
        batch_size=len(benchmark_symbols),
    )

    for required in ("SPY", "QQQ", "SOXX", "^VIX"):
        if required not in bench_daily:
            raise RuntimeError(
                f"missing critical benchmark {required}; "
                "last good JSON will be preserved"
            )

    if "^SOX" not in bench_daily:
        print("^SOX unavailable; using SOXX fallback")

    if mode == "intraday":
        session_date = now_ny.date()

        daily_map = {
            ticker: strip_current_session_daily_bar(
                df,
                session_date,
            )
            for ticker, df in daily_map.items()
        }

        bench_daily = {
            ticker: strip_current_session_daily_bar(
                df,
                session_date,
            )
            for ticker, df in bench_daily.items()
        }

    intraday_map = {}
    bench_intraday = {}

    if mode == "intraday":
        intraday_map, _ = download_batched(
            universe,
            "1d",
            "5m",
            batch_size=45,
        )

        if len(intraday_map) / len(universe) < 0.65:
            raise RuntimeError(
                f"intraday universe coverage too low "
                f"({len(intraday_map)}/{len(universe)}); "
                "last good JSON will be preserved"
            )

        active_bench = [
            "SPY",
            "QQQ",
            "^SOX" if "^SOX" in bench_daily else "SOXX",
            "^VIX",
            *SECTOR_ETFS.values(),
        ]

        bench_intraday, _ = download_batched(
            active_bench,
            "1d",
            "5m",
            batch_size=len(active_bench),
        )

    market_payload = build_market(
        mode,
        universe,
        daily_map,
        intraday_map,
        bench_daily,
        bench_intraday,
        generated,
    )

    universe_payload = None

    if mode == "eod":
        rows = []

        for ticker in universe:
            df = daily_map.get(ticker)

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

        if len(rows) / len(universe) < 0.80:
            raise RuntimeError(
                f"EOD metric coverage too low "
                f"({len(rows)}/{len(universe)}); "
                "last good JSON will be preserved"
            )

        rows.sort(key=lambda row: row["ticker"])

        dates = [
            row["price_date"]
            for row in rows
            if row.get("price_date")
        ]

        universe_payload = {
            "schema_version": 1,
            "generated_at_utc": generated.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "generated_at_hk": generated.astimezone(HK).isoformat(),
            "mode": "EOD_UNIVERSE",
            "market_date": (
                max(dates)
                if dates
                else generated.astimezone(NY).date().isoformat()
            ),
            "universe_size": len(universe),
            "valid_ticker_count": len(rows),
            "breadth": breadth_from_rows(rows),
            "definitions": {
                "sma50_slope_10d_pct":
                    "percent change in SMA50 versus 10 trading sessions earlier",
                "atr_ext_50":
                    "(close - SMA50) / ATR14",
                "extension_zone":
                    "BELOW_50MA <0; NEAR_50MA 0-<1 ATR; "
                    "ABOVE_50MA 1-<2 ATR; EXTENDED 2-<3 ATR; "
                    "VERY_EXTENDED >=3 ATR",
                "volume_dry_up_ratio":
                    "mean volume last 5 sessions / "
                    "mean volume preceding 20 sessions",
                "rs":
                    "stock return minus benchmark return over "
                    "the named session window",
            },
            "stocks": rows,
        }

    # Fail-safe: these writes happen only after all data validation succeeds.
    # If anything above raises an exception, existing last-good JSON remains.
    atomic_json_write(MARKET_PATH, market_payload)
    print(f"wrote {MARKET_PATH}")

    if universe_payload is not None:
        atomic_json_write(UNIVERSE_PATH, universe_payload)
        print(f"wrote {UNIVERSE_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"MARKET_INTELLIGENCE_FAILED: {exc}",
            file=sys.stderr,
        )
        raise
