import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from datetime import datetime

# --- Configuration ---
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
VOLUME = 0.01
CONFIDENCE_LEVEL = 0.60 # With LSTM, we can aim for 60% or higher
DEVIATION = 20

print("1. Loading LSTM V6 Model...")
try:
    model = tf.keras.models.load_model("my_ai_model_v6.h5")
    scaler = joblib.load("scaler_v6.pkl")
    print("✅ LSTM V6 Model Loaded")
except:
    print("❌ Error: Run training first.")
    quit()

if not mt5.initialize():
    quit()

def get_features():
    # Fetch Data
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None or len(rates) < 60: return None
    df = pd.DataFrame(rates)
    
    # --- Exact same calculations as V5 ---
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = pd.Series(true_range).rolling(14).mean()

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['Hour'] = df['time'].dt.hour
    df['Hour_Scaled'] = df['Hour'] / 23.0

    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['HA_Close_Lag1'] = df['HA_Close'].shift(1)
    df['Momentum'] = df['close'] - df['close'].shift(1)

    last = df.iloc[-2]
    
    features = last[['HA_Close', 'EMA_50', 'RSI', 'ATR', 'Hour_Scaled', 
                     'RSI_Lag1', 'HA_Close_Lag1', 'Momentum']].values
    
    # --- RESHAPE FOR LSTM (This is the key change) ---
    # Convert from [8] to [1, 1, 8]
    return features.reshape(1, 1, -1)

print(f"🤖SHEHAN'S AI V6 (LSTM) Started on {SYMBOL}")

while True:
    try:
        features = get_features()
        if features is not None:
            # Scale first (using 2D array logic inside scaler)
            # We need to flatten, scale, then reshape back
            features_flat = features.reshape(1, -1)
            features_scaled = scaler.transform(features_flat)
            features_lstm = features_scaled.reshape(1, 1, -1)
            
            predictions = model.predict(features_lstm, verbose=0)[0]
            action = np.argmax(predictions)
            confidence = predictions[action]
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if action == 1: signal = "BUY 🟢"
            elif action == 2: signal = "SELL 🔴"
            else: signal = "WAIT ⚪"
            
            print(f"[{timestamp}] {signal} | Conf: {confidence:.2f} | (Wait:{predictions[0]:.2f})")
            
            if confidence > CONFIDENCE_LEVEL and action != 0:
                print(f"🚀 LSTM Signal! Executing {signal}...")
                
                trade_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(SYMBOL).ask if action == 1 else mt5.symbol_info_tick(SYMBOL).bid
                
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": VOLUME,
                    "type": trade_type,
                    "price": price,
                    "deviation": DEVIATION,
                    "magic": 666666,
                    "comment": "AI V6 LSTM",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print("✅ Trade Executed!")
                    time.sleep(300)
                else:
                    print(f"❌ Failed: {res.comment}")

        time.sleep(10)

    except Exception as e:
        print("Error:", e)
        time.sleep(5)