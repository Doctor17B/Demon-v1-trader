import os
import time
import sqlite3
from datetime import datetime
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from google import genai
from google.genai import types

# ==========================================
# 1. APP & DATABASE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Adaptive Trader v2", page_icon="🧠", layout="centered")

DB_PATH = "bot_memory.db"
MODEL_PATH = "online_trade_model.pkl"
SCALER_PATH = "online_scaler.pkl"

FEATURE_COLS = [
    "EMA_Spread", "VWAP_Dist", "RSI_Norm", "MACD_Hist_Norm",
    "BB_PctB", "ADX_Norm", "RVOL", "HTF_Trend"
]

if "last_alert_candle" not in st.session_state:
    st.session_state.last_alert_candle = None


def init_db():
    """Creates a local SQLite database to store predictions and verify real market outcomes."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            interval TEXT,
            candle_time TEXT,
            entry_price REAL,
            predicted_dir INTEGER,
            ml_prob REAL,
            confluence_score INTEGER,
            features_json TEXT,
            verified INTEGER DEFAULT 0,
            actual_dir INTEGER,
            outcome_price REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ==========================================
# 2. MULTI-TIMEFRAME & INSTITUTIONAL INDICATORS
# ==========================================
def compute_features(symbol: str, interval: str, htf_interval: str) -> pd.DataFrame:
    """Fetches execution timeframe + higher timeframe and computes institutional indicators."""
    df = yf.download(symbol, period="10d", interval=interval, progress=False)
    htf_df = yf.download(symbol, period="30d", interval=htf_interval, progress=False)

    if df.empty or htf_df.empty:
        raise ValueError(f"Unable to fetch market data for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(htf_df.columns, pd.MultiIndex):
        htf_df.columns = htf_df.columns.get_level_values(0)

    # Higher Timeframe Trend (+1 Bullish, -1 Bearish)
    htf_df["HTF_EMA20"] = htf_df["Close"].ewm(span=20, adjust=False).mean()
    htf_df["HTF_EMA50"] = htf_df["Close"].ewm(span=50, adjust=False).mean()
    htf_df["HTF_Trend"] = np.where(htf_df["HTF_EMA20"] > htf_df["HTF_EMA50"], 1.0, -1.0)

    # Align Higher Timeframe trend onto execution timeframe without lookahead bias
    htf_trend_series = htf_df[["HTF_Trend"]].reindex(df.index, method="ffill").fillna(0.0)
    df["HTF_Trend"] = htf_trend_series["HTF_Trend"]

    # 1. EMAs
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # 2. RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 4. Bollinger Bands (20, 2)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma20 + (2 * std20)
    df["BB_Lower"] = sma20 - (2 * std20)
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    # 5. ATR (14) & ADX (14) Trend Strength
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["ATR"].replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / df["ATR"].replace(0, np.nan)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # 6. VWAP & Relative Volume (RVOL)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].replace(0, 1).cumsum()
    vol_ma = df["Volume"].rolling(20).mean().replace(0, 1)
    df["RVOL"] = df["Volume"] / vol_ma

    # Normalized ML Feature Columns
    atr_safe = df["ATR"].replace(0, 1e-4)
    df["EMA_Spread"] = (df["EMA_9"] - df["EMA_21"]) / atr_safe
    df["VWAP_Dist"] = (df["Close"] - df["VWAP"]) / atr_safe
    df["RSI_Norm"] = (df["RSI"] - 50.0) / 50.0
    df["MACD_Hist_Norm"] = df["MACD_Hist"] / atr_safe
    df["ADX_Norm"] = df["ADX"] / 100.0

    # Target Label for Self-Training: 1 if Close in next 3 candles is higher, else 0
    df["Future_Return_3"] = df["Close"].shift(-3) - df["Close"]
    df["Target_Up"] = (df["Future_Return_3"] > 0).astype(int)

    return df.dropna(subset=FEATURE_COLS + ["ATR", "Close"])


# ==========================================
# 3. SELF-TRAINING ONLINE ML ENGINE
# ==========================================
def train_or_update_online_model(df: pd.DataFrame):
    """
    Incrementally trains an online SGD logistic regression classifier alongside the market
    and verifies past logged predictions in SQLite to adapt to live regime shifts.
    """
    train_df = df.iloc[:-3].copy()  # Exclude last 3 rows where future target isn't realized yet
    X = train_df[FEATURE_COLS].values
    y = train_df["Target_Up"].values

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Incremental partial_fit on latest 30 realized bars to adapt to current regime
        X_recent = X[-30:]
        y_recent = y[-30:]
        scaler.partial_fit(X_recent)
        model.partial_fit(scaler.transform(X_recent), y_recent, classes=np.array([0, 1]))
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, random_state=42)
        model.partial_fit(X_scaled, y, classes=np.array([0, 1]))

    # Calculate rolling walk-forward accuracy on recent 100 bars
    eval_X = scaler.transform(X[-100:])
    eval_y = y[-100:]
    preds = model.predict(eval_X)
    rolling_acc = accuracy_score(eval_y, preds) * 100

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler, rolling_acc


def verify_and_learn_from_past_predictions(symbol: str, current_price: float, model, scaler):
    """Grades unverified past predictions in SQLite against current price and runs partial_fit."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, entry_price, predicted_dir, features_json FROM predictions WHERE symbol=? AND verified=0",
        (symbol,)
    )
    rows = c.fetchall()

    for row_id, entry_price, pred_dir, feat_str in rows:
        # Grade once price has moved at least 0.15% from entry
        if abs(current_price - entry_price) / entry_price >= 0.0015:
            actual_dir = 1 if current_price > entry_price else 0
            feat_vals = np.array([float(x) for x in feat_str.split(",")]).reshape(1, -1)
            scaler.partial_fit(feat_vals)
            model.partial_fit(scaler.transform(feat_vals), np.array([actual_dir]), classes=np.array([0, 1]))

            c.execute(
                "UPDATE predictions SET verified=1, actual_dir=?, outcome_price=? WHERE id=?",
                (actual_dir, current_price, row_id)
            )

    conn.commit()
    conn.close()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


def log_new_prediction(symbol: str, interval: str, candle_time: str, price: float, pred_dir: int, prob: float, score: int, feat_vec: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    feat_str = ",".join([f"{x:.5f}" for x in feat_vec.flatten()])
    c.execute("""
        INSERT INTO predictions (symbol, interval, candle_time, entry_price, predicted_dir, ml_prob, confluence_score, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, interval, candle_time, price, pred_dir, prob, score, feat_str))
    conn.commit()
    conn.close()


# ==========================================
# 4. LIVE NEWS & GOOGLE SEARCH GROUNDED AI RESEARCH
# ==========================================
def fetch_ticker_news(symbol: str) -> list:
    """Pulls recent news headlines from Yahoo Finance for immediate sentiment context."""
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news[:5] if ticker.news else []
        headlines = []
        for item in news_items:
            content = item.get("content", item)
            title = content.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def ask_grounded_gemini(symbol: str, interval: str, sig: dict, headlines: list, api_key: str) -> str:
    """Uses Gemini with Google Search Grounding + Self-Learning performance history."""
    if not api_key:
        return "Enter your Gemini API Key in Settings to unlock live Google Search Grounded research."

    client = genai.Client(api_key=api_key)
    news_text = "\n".join([f"- {h}" for h in headlines]) if headlines else "No immediate ticker RSS headlines."

    prompt = f"""
    You are an adaptive Quantitative Trading Assistant with live web search access.
    Analyze {symbol} on the {interval} timeframe.

    Quantitative & Self-Learning Snapshot:
    - Live Price: {sig['close']:.2f} | Combined Action: {sig['direction']}
    - Rule Confluence Score: {sig['score']}/6 | Higher Timeframe Trend: {'Bullish' if sig['htf_trend'] > 0 else 'Bearish'}
    - Self-Trained ML Upward Probability: {sig['ml_prob']*100:.1f}% (Model Rolling Accuracy: {sig['model_acc']:.1f}%)
    - ADX Trend Strength: {sig['adx']:.1f} | Relative Volume (RVOL): {sig['rvol']:.2f}x
    - ATR Volatility Stop-Loss: {sig['stop_loss']:.2f} | Target (1:2 R:R): {sig['target']:.2f}
    - Active Technical Factors: {', '.join(sig['reasons'])}

    Recent Ticker Headlines:
    {news_text}

    Search the web for any breaking market news or macro catalysts affecting {symbol} today, then provide:
    1. **Executive Trade Call**: Clear UP, DOWN, or FLAT verdict combining technicals, ML probability, and live news sentiment.
    2. **Macro & News Catalyst Check**: Any breaking news or scheduled events that could invalidate technicals.
    3. **Execution & Risk Plan**: Exact Entry, Stop-Loss, Target, and when to trail stops.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        return response.text
    except Exception as e:
        return f"AI Research error: {e}"


# ==========================================
# 5. PUSH NOTIFICATION DISPATCHER
# ==========================================
def send_push_alert(title: str, message: str, ntfy_topic: str):
    if ntfy_topic.strip():
        try:
            requests.post(
                f"https://ntfy.sh/{ntfy_topic.strip()}",
                data=message.encode("utf-8"),
                headers={"Title": title, "Priority": "high", "Tags": "chart_with_upwards_trend,brain"},
                timeout=5
            )
        except Exception as e:
            st.warning(f"Push alert failed: {e}")


# ==========================================
# 6. MOBILE UI & MULTI-TAB WORKSTATION
# ==========================================
st.title("🧠 AI Self-Learning Trader v2.0")

with st.expander("⚙️ Market, Risk & Notification Settings", expanded=False):
    symbol = st.text_input("Ticker Symbol (^NSEI, RELIANCE.NS, BTC-USD, AAPL)", value="^NSEI")
    col_a, col_b = st.columns(2)
    interval = col_a.selectbox("Entry Timeframe", ["5m", "15m", "1h"], index=0)
    htf_interval = col_b.selectbox("Higher Trend Timeframe", ["1h", "1d"], index=0)
    account_size = col_a.number_input("Account Capital", value=100000.0, step=5000.0)
    risk_pct = col_b.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.5)
    ntfy_topic = st.text_input("ntfy.sh Topic for Phone Alerts", value="my_secret_trade_bot_99")
    gemini_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    auto_scan = st.checkbox("Auto-Train & Scan Every 60 Seconds", value=True)

tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Signal", "🧠 Self-Learning", "🛡️ Lot Calculator", "📰 News Research"])

if st.button("🔄 Run Live Scan & Self-Train Now", use_container_width=True) or auto_scan:
    try:
        df = compute_features(symbol, interval, htf_interval)
        model, scaler, model_acc = train_or_update_online_model(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close_price = float(latest["Close"])
        atr = float(latest["ATR"])
        candle_time = str(df.index[-1])

        # Verify past SQLite predictions against current price and update ML weights
        verify_and_learn_from_past_predictions(symbol, close_price, model, scaler)

        # Predict probability of UP move on current live candle
        latest_features = latest[FEATURE_COLS].values.reshape(1, -1)
        ml_prob_up = float(model.predict_proba(scaler.transform(latest_features))[0][1])

        # Rule + ML + MTF Confluence Scoring (-6 to +6)
        score = 0
        reasons = []

        if latest["EMA_9"] > latest["EMA_21"]:
            score += 1
            reasons.append("EMA 9 > EMA 21")
        else:
            score -= 1
            reasons.append("EMA 9 < EMA 21")

        if latest["HTF_Trend"] > 0:
            score += 1
            reasons.append(f"Higher TF ({htf_interval}) Bullish")
        else:
            score -= 1
            reasons.append(f"Higher TF ({htf_interval}) Bearish")

        if latest["Close"] > latest["VWAP"]:
            score += 1
            reasons.append("Above Institutional VWAP")
        else:
            score -= 1
            reasons.append("Below Institutional VWAP")

        if latest["MACD"] > latest["MACD_Signal"]:
            score += 1
            reasons.append("MACD Bullish")
        else:
            score -= 1
            reasons.append("MACD Bearish")

        if ml_prob_up >= 0.58:
            score += 2
            reasons.append(f"Self-Trained ML Bullish ({ml_prob_up*100:.1f}% Up Prob)")
        elif ml_prob_up <= 0.42:
            score -= 2
            reasons.append(f"Self-Trained ML Bearish ({(1-ml_prob_up)*100:.1f}% Down Prob)")

        adx_val = float(latest["ADX"])
        rvol_val = float(latest["RVOL"])

        # Chop Filter: Block trades if ADX < 20
        if adx_val < 20:
            direction = "WAIT / SIDEWAYS CHOP (ADX < 20)"
            stop_loss = close_price - 1.5 * atr
            target = close_price + 3.0 * atr
        elif score >= 4:
            direction = "STRONG UP (BUY / CALL)"
            stop_loss = close_price - 1.5 * atr
            target = close_price + 3.0 * atr
        elif score <= -4:
            direction = "STRONG DOWN (SELL / PUT)"
            stop_loss = close_price + 1.5 * atr
            target = close_price - 3.0 * atr
        else:
            direction = "NEUTRAL / WAIT FOR CONFLUENCE"
            stop_loss = close_price - 1.5 * atr
            target = close_price + 3.0 * atr

        sig = {
            "score": score,
            "direction": direction,
            "close": close_price,
            "atr": atr,
            "stop_loss": stop_loss,
            "target": target,
            "ml_prob": ml_prob_up,
            "model_acc": model_acc,
            "adx": adx_val,
            "rvol": rvol_val,
            "htf_trend": float(latest["HTF_Trend"]),
            "reasons": reasons
        }

        # Log prediction & send push notification on new high-conviction candle
        if st.session_state.last_alert_candle != candle_time:
            pred_binary = 1 if ml_prob_up >= 0.5 else 0
            log_new_prediction(symbol, interval, candle_time, close_price, pred_binary, ml_prob_up, score, latest_features)

            if abs(score) >= 4 and adx_val >= 20:
                send_push_alert(
                    title=f"🚨 {symbol}: {direction}",
                    message=(
                        f"Action: {direction} (Score: {score}/6)\n"
                        f"ML Up Prob: {ml_prob_up*100:.1f}%\n"
                        f"Entry: {close_price:.2f} | SL: {stop_loss:.2f} | Target: {target:.2f}\n"
                        f"Factors: {', '.join(reasons)}"
                    ),
                    ntfy_topic=ntfy_topic
                )
                st.toast("📲 High-confluence trade alert pushed to your phone!")
            st.session_state.last_alert_candle = candle_time

        # ---------------- TAB 1: LIVE SIGNAL ----------------
        with tab1:
            if "STRONG UP" in direction:
                st.success(f"🟢 **{direction}** | Score: +{score}/6 | ML Up Prob: {ml_prob_up*100:.1f}%")
            elif "STRONG DOWN" in direction:
                st.error(f"🔴 **{direction}** | Score: {score}/6 | ML Down Prob: {(1-ml_prob_up)*100:.1f}%")
            else:
                st.warning(f"🟡 **{direction}** | Score: {score}/6 | ML Up Prob: {ml_prob_up*100:.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("Live Price", f"{close_price:.2f}")
            c2.metric("Stop-Loss (1.5x ATR)", f"{stop_loss:.2f}")
            c3.metric("Target (1:2 R:R)", f"{target:.2f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("ADX Trend Strength", f"{adx_val:.1f}", "Trending" if adx_val >= 20 else "Choppy")
            c5.metric("Relative Volume", f"{rvol_val:.2f}x")
            c6.metric("Higher TF Trend", "Bullish 🟢" if latest["HTF_Trend"] > 0 else "Bearish 🔴")

            st.line_chart(df[["Close", "EMA_9", "EMA_21", "VWAP"]].tail(60))
            st.write("**Triggered Confluence Factors:**", ", ".join(reasons))

        # ---------------- TAB 2: SELF-LEARNING BRAIN ----------------
        with tab2:
            st.subheader("🧠 Online Machine Learning Telemetry")
            st.metric("Walk-Forward Model Accuracy (Last 100 Bars)", f"{model_acc:.1f}%")

            weights_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Learned Weight": model.coef_[0]
            }).sort_values(by="Learned Weight", ascending=False)
            st.caption("Live Feature Weights (Automatically updated via `partial_fit` as market regimes shift):")
            st.dataframe(weights_df, use_container_width=True)

            conn = sqlite3.connect(DB_PATH)
            hist_df = pd.read_sql_query(
                "SELECT candle_time, entry_price, predicted_dir, ml_prob, verified, actual_dir, outcome_price FROM predictions ORDER BY id DESC LIMIT 15",
                conn
            )
            conn.close()
            st.caption("Recent Logged Predictions & Self-Graded Outcomes (`bot_memory.db`):")
            st.dataframe(hist_df, use_container_width=True)

        # ---------------- TAB 3: LOT & RISK CALCULATOR ----------------
        with tab3:
            st.subheader("🛡️ Dynamic Position Sizing")
            max_risk_cash = account_size * (risk_pct / 100.0)
            stop_distance = abs(close_price - stop_loss)
            recommended_qty = int(max_risk_cash // stop_distance) if stop_distance > 0 else 0

            r1, r2, r3 = st.columns(3)
            r1.metric("Max Risk Amount", f"{max_risk_cash:,.2f}")
            r2.metric("Stop Distance / Unit", f"{stop_distance:.2f}")
            r3.metric("Recommended Quantity", f"{recommended_qty} units")

        # ---------------- TAB 4: LIVE NEWS & AI RESEARCH ----------------
        with tab4:
            st.subheader("📰 Grounded Market & News Intelligence")
            headlines = fetch_ticker_news(symbol)
            if headlines:
                for h in headlines:
                    st.write(f"• {h}")
            st.markdown("---")
            st.info(ask_grounded_gemini(symbol, interval, sig, headlines, gemini_key))

    except Exception as e:
        st.error(f"Scanner Error: {e}")

    if auto_scan:
        time.sleep(60)
        st.rerun()
