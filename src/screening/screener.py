"""
Stock screening module modified for Qullamaggie Momentum & Consolidation Setup.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.storage import StockDatabase
from .indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    detect_volume_spike,
    find_swing_lows
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_momentum_score(price_df: pd.DataFrame) -> float:
    """
    【Qullamaggie 改裝 1】：計算動能評分
    不看估值，只看這隻股票夠不夠強！
    """
    if price_df.empty or len(price_df) < 50:
        return 0.0

    score = 0.0
    close_prices = price_df['Close']
    current_price = close_prices.iloc[-1]

    # 1. 趨勢檢查：股價必須在 10日、20日、50日、200日均線之上 (強勢股基本條件)
    sma_10 = close_prices.rolling(window=10).mean().iloc[-1]
    sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
    sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
    sma_200 = close_prices.rolling(window=200).mean().iloc[-1]

    if current_price > sma_10: score += 15
    if current_price > sma_20: score += 15
    if current_price > sma_50: score += 20
    if current_price > sma_200: score += 20

    # 2. 過去 1 到 3 個月有沒有爆發過？
    price_50_days_ago = close_prices.iloc[-50]
    gain_pct = ((current_price - price_50_days_ago) / price_50_days_ago) * 100
    if gain_pct >= 30:
        score += 30

    return min(score, 100.0)


def check_is_consolidating(price_df: pd.DataFrame) -> float:
    """
    【Qullamaggie 改裝 2】：檢查是否在高位窄幅整理（Flag 形態）
    回傳一個 0-100 的分數，分數越高代表橫盤震盪越窄。
    """
    if len(price_df) < 15:
        return 0.0
    
    # 拿過去 10 個交易日的最高價和最低價
    recent_data = price_df.iloc[-10:]
    highest = recent_data['High'].max()
    lowest = recent_data['Low'].min()
    
    # 計算這 10 天的震盪幅度波動百分比
    volatility = ((highest - lowest) / lowest) * 100
    
    # Qullamaggie 喜歡波動小於 10% 甚至 5% 縮量橫盤的股票
    if volatility <= 5.0:
        return 100.0  # 極度緊湊的整理，隨時突破！
    elif volatility <= 10.0:
        return 70.0   # 標準的 Flag 整理
    elif volatility <= 15.0:
        return 40.0   # 震盪稍微偏大
    else:
        return 0.0    # 震盪太大，不符合形態


def screen_candidates(
    db: StockDatabase,
    tickers_list: List[str],
    value_weight: float = 0.7,
    support_weight: float = 0.3,
    min_data_days: int = 200
) -> pd.DataFrame:
    """
    Main screening function that combines momentum analysis with consolidation detection.
    """
    if not tickers_list:
        logger.warning("Empty ticker list provided")
        return pd.DataFrame()

    logger.info(f"Screening {len(tickers_list)} candidates...")
    results = []

    for ticker in tickers_list:
        try:
            logger.debug(f"Processing {ticker}...")

            # Get fundamental data (保留用來拿公司名字同板塊)
            fundamentals = db.get_latest_fundamentals(ticker)
            if not fundamentals:
                logger.warning(f"No fundamentals found for {ticker}")
                continue

            # Get price history
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=min_data_days + 100)

            price_history = db.get_price_history(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if price_history.empty or len(price_history) < min_data_days:
                logger.warning(f"Insufficient price data for {ticker}: {len(price_history)} days")
                continue

            # === 【Qullamaggie 動能選股改裝】 ===
            
            # 1. 提取當前股價
            current_price = fundamentals.get('current_price')
            if current_price is None or current_price <= 0:
                if not price_history.empty:
                    current_price = float(price_history['Close'].iloc[-1])
                else:
                    logger.warning(f"Invalid current price for {ticker}")
                    continue

            # 2. 計算動能分
            value_score = calculate_momentum_score(price_history)

            # 3. 計算整理緊湊度分
            support_score = check_is_consolidating(price_history)

            # 4. 計算 RSI (保留給報表參考)
            rsi = None
            if len(price_history) >= 15:
                rsi_series = calculate_rsi(price_history['Close'], period=14)
                if not rsi_series.isna().all():
                    rsi = float(rsi_series.iloc[-1])

            # 5. 檢查當天有沒有爆量？
            volume_spike = False
            if 'Volume' in price_history.columns and len(price_history) >= 20:
                current_volume = price_history['Volume'].iloc[-1]
                avg_volume = price_history['Volume'].iloc[-21:-1].mean()
                if current_volume > (avg_volume * 1.5):
                    volume_spike = True

            # 6. 計算最終信號得分
            buy_signal = (value_score * 0.5) + (support_score * 0.5)
            if volume_spike and support_score > 50:
                buy_signal += 10  # 爆量突破前兆加分
            buy_signal = min(buy_signal, 100.0)

            # 7. 用 10 日線當作移動支撐
            nearest_support = None
            if len(price_history) >= 10:
                nearest_support = float(price_history['Close'].rolling(window=10).mean().iloc[-1])

            # Compile results
            results.append({
                'ticker': ticker,
                'name': fundamentals.get('name', ticker),
                'sector': fundamentals.get('sector', 'Unknown'),
                'current_price': current_price,
                'value_score': value_score,
                'support_score': support_score,
                'buy_signal': round(buy_signal, 2),
                'rsi': round(rsi, 2) if rsi is not None else None,
                'nearest_support': round(nearest_support, 2) if nearest_support else None,
                'pe_ratio': fundamentals.get('pe_ratio'),
                'pb_ratio': fundamentals.get('pb_ratio'),
                'data_points': len(price_history)
            })

            logger.info(
                f"{ticker}: Buy Signal={buy_signal:.1f} "
                f"(Momentum={value_score:.1f}, Tightness={support_score:.1f})"
            )

        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
            continue

    if not results:
        logger.warning("No valid screening results")
        return pd.DataFrame()

    # Create DataFrame and sort by buy signal
    df = pd.DataFrame(results)
    df = df.sort_values('buy_signal', ascending=False).reset_index(drop=True)
    logger.info(f"Successfully screened {len(df)} candidates")

    return df
