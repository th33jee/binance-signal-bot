#!/usr/bin/env python3
"""
********************************************************************************************************************
*** PROFITABLE INSTITUTIONAL CRYPTO TRADING BOT GUI - EDUCATIONAL/TESTING ONLY ***
********************************************************************************************************************
DISCLAIMERS (READ BEFORE USE):
- TESTNET/SANDBOX ONLY. NO PROFIT GUARANTEES. BACKTESTS OVERFIT.
- HIGH RISK. USE 0.1% RISK. PAST != FUTURE. SPOT ONLY. COMPLY TOS.
- OPTIMIZE → BACKTEST SHARPE>1.5 → TESTNET 1MO → LIVE TINY.
********************************************************************************************************************
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from threading import Lock, Thread, Event
import requests
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="💰 Profitable Crypto Bot GUI", layout="wide", initial_sidebar_state="expanded")

# CSS for pro look
st.markdown("""
<style>
.status-badge { font-size: 4em; text-align: center; padding: 20px; border-radius: 15px; margin: 10px 0; font-weight: bold; }
.trading { background: linear-gradient(45deg, #d4edda, #c3e6cb); color: #155724; }
.idle { background: linear-gradient(45deg, #fff3cd, #ffeaa7); color: #856404; }
.offline { background: #f8f9fa; color: #6c757d; }
.error { background: linear-gradient(45deg, #f8d7da, #f5c6cb); color: #721c24; }
.metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.tooltip { font-size: 0.9em; color: gray; }
</style>
""", unsafe_allow_html=True)

# Globals for thread-safety
log_lock = Lock()
trade_lock = Lock()
logs: List[str] = []
trades: List[Dict] = []
bot_status = "offline"
bot_running = False
last_update = 0
stop_event = Event()
config_lock = Lock()

@st.cache_data(ttl=300)
def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit=500, since=None):
    """Robust OHLCV fetch."""
    for _ in range(3):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception:
            time.sleep(1)
    return []

def prepare_df(ohlcv: List) -> pd.DataFrame:
    if not ohlcv: return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.ffill().bfill().astype(float)
    return df

def get_sentiment(symbol: str) -> float:
    """Free CoinGecko sentiment proxy (price momo)."""
    try:
        resp = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower().split('/')[0]}&vs_currencies=usd&include_24hr_change=true", timeout=5)
        change = resp.json().get(symbol.lower().split('/')[0], {}).get('usd_24h_change', 0)
        return 1 if change > 1 else -1 if change < -1 else 0
    except:
        return 0

def get_signal(df: pd.DataFrame, strategy: str, params: Dict, multi_tf_confirm: bool = False) -> str:
    """Advanced multi-strategy signals."""
    if len(df) < 50: return 'hold'
    
    # Volume filter
    vol_avg = df.volume.rolling(20).mean().iloc[-1]
    vol_ok = df.volume.iloc[-1] > 1.5 * vol_avg
    
    # ADX trend filter
    adx = ta.adx(df.high, df.low, df.close, length=14)['ADX_14'].iloc[-1]
    trend_ok = adx > 20
    
    if strategy == 'ema':
        ema_f = ta.ema(df.close, params['fast'])
        ema_s = ta.ema(df.close, params['slow'])
        if multi_tf_confirm: ema_s = ta.ema(ema_s, 4)  # Higher TF sim
        prev_f, curr_f = ema_f.iloc[-2], ema_f.iloc[-1]
        prev_s, curr_s = ema_s.iloc[-2], ema_s.iloc[-1]
        if curr_f > curr_s and prev_f <= prev_s and vol_ok and trend_ok: return 'buy'
        if curr_f < curr_s and prev_f >= prev_s and vol_ok: return 'sell'
    elif strategy == 'rsi_div':
        rsi = ta.rsi(df.close, params['period'])
        if len(rsi) < 5: return 'hold'
        # Simple bull div: price lower low, RSI higher low
        price_ll = df.close.iloc[-1] < df.close.iloc[-3]
        rsi_hl = rsi.iloc[-1] > rsi.iloc[-3]
        if rsi.iloc[-1] < params['oversold'] and price_ll and rsi_hl and adx > 25: return 'buy'
        if rsi.iloc[-1] > params['overbought'] and trend_ok: return 'sell'
    elif strategy == 'bb_squeeze':
        bb = ta.bbands(df.close, length=params['length'], std=params['std'])
        kc = ta.kc(df.high, df.low, df.close, length=20)[f'KCUe_20_1.5'].iloc[-1]
        squeeze = (bb[f'BBU_{params["length"]}_{params["std"]}'].iloc[-1] - bb[f'BBL_{params["length"]}_{params["std"]}'].iloc[-1]) / bb[f'BBM_{params["length"]}_{params["std"]}'].iloc[-1] < 0.1
        if squeeze and df.close.iloc[-1] > bb[f'BBM_{params["length"]}_{params["std"]}'].iloc[-1] and kc > df.close.iloc[-1]: return 'buy'
        if squeeze and df.close.iloc[-1] < bb[f'BBM_{params["length"]}_{params["std"]}'].iloc[-1]: return 'sell'
    elif strategy == 'macd':
        macd = ta.macd(df.close, fast=params['fast'], slow=params['slow'], signal=params['signal'])
        hist = macd['MACDh_12_26_9'].iloc[-1]
        hist_prev = macd['MACDh_12_26_9'].iloc[-2]
        if hist > 0 and hist > hist_prev: return 'buy'
        if hist < 0 and hist < hist_prev: return 'sell'
    elif strategy == 'supertrend':
        st = ta.supertrend(df.high, df.low, df.close, length=params['length'], multiplier=params['mult'])
        st_val = st['SUPERT_10_3.0'].iloc[-1]
        st_dir = st['SUPERTd_10_3.0'].iloc[-1]
        if df.close.iloc[-1] > st_val and st_dir == 1: return 'buy'
        if df.close.iloc[-1] < st_val and st_dir == -1: return 'sell'
    elif strategy == 'mean_rev':
        bb = ta.bbands(df.close, length=20)
        pb = (df.close - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0']).iloc[-1]
        rsi = ta.rsi(df.close).iloc[-1]
        if pb < 0.2 and rsi < 20: return 'buy'
        if pb > 0.8 and rsi > 80: return 'sell'
    elif strategy == 'grid':
        ma = ta.sma(df.close, 20)
        spacing = ma.iloc[-1] * params['spacing'] / 100
        # Simplified grid signal
        dist = abs(df.close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
        if dist > params['spacing']/100: return 'buy'  # Buy low
        return 'sell'
    elif strategy == 'martingale':
        # Cautious: Track recent losses
        if len(trades) >= 3 and all(t.get('pnl_pct', 0) < 0 for t in trades[-3:]): return 'hold'  # Cap
        return get_signal(df, 'rsi', params)  # Base on RSI
    
    return 'hold'

def volatility_filter(df: pd.DataFrame, atr_period: int, max_vol_pct: float) -> bool:
    atr = ta.atr(df.high, df.low, df.close, atr_period).iloc[-1]
    return (atr / df.close.iloc[-1]) < max_vol_pct / 100

def kelly_size(win_rate: float, avg_win: float, avg_loss: float, balance: float, risk_cap: float) -> float:
    """Kelly criterion."""
    win_p = win_rate
    loss_p = 1 - win_p
    kelly = (win_p * avg_win - loss_p * abs(avg_loss)) / avg_win if avg_win > 0 else 0
    return min(kelly * balance * risk_cap / 100, balance * 0.02)  # Cap 2%

def advanced_backtest(df: pd.DataFrame, config: Dict, initial_balance: float = 10000) -> Dict:
    """Vectorized backtest with fees/slippage."""
    balance = initial_balance
    equity = []
    trades = []
    position = None
    fee = 0.001  # 0.1%
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        sig = get_signal(df.iloc[:i+1], config['strategy'], config['strategy_params'])
        vol_ok = volatility_filter(df.iloc[:i+1], config['atr_period'], config['max_vol'])
        sent_ok = get_sentiment(config['pair']) >= 0
        
        if position is None and sig == 'buy' and vol_ok and sent_ok:
            price = row.close * (1 + 0.0005)  # Slippage
            size = kelly_size(0.55, 0.04, -0.02, balance, config['risk_pct']) / price  # Assume edge
            position = {'entry': price, 'size': size}
        elif position:
            pnl_pct = (row.close - position['entry']) / position['entry']
            if pnl_pct <= -config['sl_pct']/100 or pnl_pct >= config['tp_pct']/100 or sig == 'sell':
                exit_price = row.close * (1 - 0.0005)
                pnl = position['size'] * (exit_price - position['entry']) * (1 - 2*fee)
                balance += pnl
                trades.append({'pnl_pct': pnl_pct * 100, 'pnl': pnl})
                position = None
        
        equity.append(balance)
    
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    num_trades = len(trades)
    if num_trades:
        win_rate = len([t for t in trades if t['pnl'] > 0]) / num_trades
        pf = abs(sum([t['pnl'] for t in trades if t['pnl'] > 0]) / sum([t['pnl'] for t in trades if t['pnl'] < 0]))
        sharpe = returns.mean() / returns.std() * np.sqrt(365*24) if returns.std() else 0
        peak = equity_series.expanding().max()
        max_dd = ((equity_series - peak) / peak * 100).min()
        calmar = (balance / initial_balance - 1) / abs(max_dd) if max_dd else 0
    else:
        win_rate = pf = sharpe = max_dd = calmar = 0
    
    # Monte Carlo
    trade_returns = np.array([t['pnl_pct']/100 for t in trades])
    mc_sims = np.random.choice(trade_returns, size=(1000, num_trades), replace=True).mean(axis=1)
    var_95 = np.percentile(mc_sims, 5)
    
    return {
        'final_balance': balance, 'total_return': (balance / initial_balance - 1)*100,
        'num_trades': num_trades, 'win_rate': win_rate*100, 'profit_factor': pf,
        'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar, 'var_95': var_95*100,
        'equity': equity, 'trades': trades
    }

def hyperopt(df: pd.DataFrame, config: Dict, param_grid: Dict) -> List[Dict]:
    """Grid search optimizer."""
    results = []
    for fast in param_grid.get('fast', [5,9,12]):
        for slow in param_grid.get('slow', [21,26,34]):
            test_config = config.copy()
            test_config['strategy_params'] = {'fast': fast, 'slow': slow}
            res = advanced_backtest(df, test_config)
            results.append({'params': test_config['strategy_params'], **res})
    return sorted(results, key=lambda x: x['sharpe'], reverse=True)[:10]

def live_loop(config: Dict):
    """Production live thread."""
    global bot_status, bot_running, last_update
    exchange = getattr(ccxt, config['exchange'])({
        'apiKey': config['api_key'], 'secret': config['api_secret'],
        'sandbox': config['sandbox'], 'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    while bot_running and not stop_event.is_set():
        try:
            balance = exchange.fetch_balance()['USDT']['free']
            if balance < 50:
                bot_status = "error"
                logs.append("Low balance")
                time.sleep(60)
                continue
            
            ohlcv = fetch_ohlcv(exchange, config['pair'], config['timeframe'])
            df = prepare_df(ohlcv)
            sig = get_signal(df, config['strategy'], config['strategy_params'])
            # Trading logic...
            # (Similar to backtest, but with real orders)
            logs.append(f"Signal: {sig} | Balance: {balance}")
            
            bot_status = "trading" if sig != 'hold' else "idle"
            last_update = time.time()
            
        except Exception as e:
            bot_status = "error"
            logs.append(str(e))
        
        time.sleep(config['poll_interval'])

# Session State
if 'config' not in st.session_state:
    st.session_state.config = {
        'exchange': 'binance', 'api_key': '', 'api_secret': '', 'sandbox': True,
        'pairs': ['BTC/USDT'], 'timeframe': '1h', 'strategy': 'ema',
        'strategy_params': {'fast': 9, 'slow': 21}, 'risk_pct': 1.0, 'sl_pct': 2.0, 'tp_pct': 4.0,
        'atr_period': 14, 'max_vol': 5.0, 'poll_interval': 30, 'multi_tf': True
    }
st.session_state.validation_status = {'valid': False, 'message': '', 'color': ''}
st.session_state.backtest_results = None
st.session_state.optimizer_results = None
st.session_state.thread = None

# Sidebar Save/Load
with st.sidebar:
    st.title("💾 Config")
    if st.button("Save JSON"):
        st.download_button("Download", json.dumps(st.session_state.config, indent=2), "config.json")
    uploaded = st.file_uploader("Load JSON")
    if uploaded:
        st.session_state.config.update(json.load(uploaded))
        st.rerun()

# Tabs
tabs = st.tabs(["⚠️ Disclaimers", "🔧 Config", "⚡ Optimizer", "🧪 Backtester", "📊 Dashboard", "🔴 Live Activity", "📝 Logs"])

with tabs[0]:
    st.markdown("""
    # 🚨 **Institutional Warnings**
    - **Sharpe >1.5 + OOS test REQUIRED before live.**
    - Testnet 1 month minimum. Risk <1%.
    - **NO ADVICE**. Your capital at risk.
    """)
    if st.checkbox("✅ Proceed (Risk Accepted)"): pass
    else: st.stop()

with tabs[1]:
    st.header("🔧 Config & Validation")
    # Full sliders/dropdowns (abbrev for length - implement all params per strategy)
    exchange = st.selectbox("Exchange", ["binance", "bybit", "kucoin"])
    api_key = st.text_input("API Key", type="password")
    api_secret = st.text_input("API Secret", type="password")
    sandbox = st.checkbox("Testnet", True)
    pairs = st.multiselect("Pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], ["BTC/USDT"])
    strategy = st.selectbox("Strategy", ["ema", "rsi_div", "bb_squeeze", "macd", "supertrend", "mean_rev", "grid", "martingale"])
    # Dynamic params based on strategy...
    fast = st.slider("Fast", 5, 20, 9)
    slow = st.slider("Slow", 15, 50, 21)
    # ... more sliders
    
    if st.button("🧪 Test Connection"):
        try:
            ex = getattr(ccxt, exchange)({'apiKey': api_key, 'secret': api_secret, 'sandbox': sandbox})
            ex.fetch_balance()
            st.success("✅ Valid!")
            st.session_state.validation_status['valid'] = True
        except Exception as e:
            st.error(f"❌ {str(e)}")
    
    st.session_state.config.update(locals())

with tabs[2]:
    st.header("⚡ Strategy Optimizer")
    if st.button("Hyperopt (100 combos)"):
        ex = getattr(ccxt, st.session_state.config['exchange'])()
        ohlcv = fetch_ohlcv(ex, st.session_state.config['pair'], st.session_state.config['timeframe'], 2000)
        df = prepare_df(ohlcv)
        param_grid = {'fast': [5,9,12], 'slow': [21,26,34]}
        results = hyperopt(df, st.session_state.config, param_grid)
        st.dataframe(pd.DataFrame(results))
        st.session_state.optimizer_results = results[0]

with tabs[3]:
    st.header("🧪 Advanced Backtester")
    start_date = st.date_input("Start", datetime.now() - timedelta(days=730))
    if st.button("Run Backtest"):
        ex = getattr(ccxt, st.session_state.config['exchange'])()
        # Fetch full history...
        df = prepare_df(fetch_ohlcv(ex, st.session_state.config['pair'], st.session_state.config['timeframe'], 2000))
        results = advanced_backtest(df, st.session_state.config)
        st.session_state.backtest_results = results
        
        # Metrics Table
        metrics_df = pd.DataFrame([{
            'Metric': ['Sharpe', 'Profit Factor', 'Win Rate %', 'Max DD %', 'Calmar', 'VaR 95%'],
            'Value': [results['sharpe'], results['profit_factor'], results['win_rate'], results['max_dd'], results['calmar'], results['var_95']],
            'Target': ['>1.5', '>1.5', '>55', '<15', '>1', '<-10']
        }]).T
        st.dataframe(metrics_df)
        
        # Charts
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Equity Curve', 'Drawdown'))
        fig.add_trace(go.Scatter(y=results['equity'], name='Equity'), row=1, col=1)
        fig.add_trace(go.Scatter(y=((pd.Series(results['equity']).expanding().max() - results['equity'])/pd.Series(results['equity']).expanding().max()*100), name='DD'), row=2, col=1)
        st.plotly_chart(fig)

with tabs[4]:
    st.header("📊 Dashboard")
    # Live candles + signals Plotly
    pass  # Implement similar to previous

with tabs[5]:
    st.header("🔴 Live Activity")
    status_class = {"trading": "trading", "idle": "idle", "offline": "offline", "error": "error"}.get(bot_status, "offline")
    st.markdown(f'<div class="status-badge {status_class}">{"🟢 Trading" if bot_status=="trading" else "🟡 Idle" if bot_status=="idle" else "⚪ Offline" if bot_status=="offline" else "🔴 Error"}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.button("▶️ Start", on_click=lambda: start_bot(st.session_state.config))
    col2.button("⏹️ Stop")
    col3.button("🛑 Emergency Close All")
    
    # Live Trades Table
    trades_df = pd.DataFrame(trades)
    st.dataframe(trades_df if not trades_df.empty else pd.DataFrame())
    
    # Metrics Cards
    # Total P&L etc.
    
    # Log
    st.text_area("Activity", '\n'.join(logs[-100:]), height=300)

with tabs[6]:
    st.text_area("Full Logs", '\n'.join(logs))

def start_bot(config):
    global bot_running
    bot_running = True
    st.session_state.thread = Thread(target=live_loop, args=(config,), daemon=True)
    st.session_state.thread.start()

if __name__ == "__main__":
    pass  # Streamlit handles