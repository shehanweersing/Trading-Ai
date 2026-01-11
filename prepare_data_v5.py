import pandas as pd
import numpy as np

# Note: Removed pandas_ta import to avoid errors. 
# We will use manual calculation for ATR below.

print("1. Loading Data...")
df = pd.read_csv('xauusd_data.csv')

# --- Strict Filtering: Use only last 50,000 candles ---
df = df.tail(50000).reset_index(drop=True)

print("2. Calculating Advanced Features...")

# 1. Standard Indicators
df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

# 2. RSI Calculation
def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
df['RSI'] = get_rsi(df['close'])

# 3. ATR (Volatility) - Manual Calculation
# (No external library needed)
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df['ATR'] = pd.Series(true_range).rolling(14).mean()

# 4. TIME Features
df['time'] = pd.to_datetime(df['time'])
df['Hour'] = df['time'].dt.hour
df['Hour_Scaled'] = df['Hour'] / 23.0

# 5. Lag Features
df['RSI_Lag1'] = df['RSI'].shift(1)
df['HA_Close_Lag1'] = df['HA_Close'].shift(1)
df['Momentum'] = df['close'] - df['close'].shift(1)

# --- Target Logic (3 Classes) ---
threshold = 0.30 
conditions = [
    (df['close'].shift(-1) > df['close'] + threshold),  # Buy (1)
    (df['close'].shift(-1) < df['close'] - threshold)   # Sell (2)
]
choices = [1, 2]
df['Target'] = np.select(conditions, choices, default=0)

df = df.dropna()

final_df = df[['HA_Close', 'EMA_50', 'RSI', 'ATR', 'Hour_Scaled', 
               'RSI_Lag1', 'HA_Close_Lag1', 'Momentum', 'Target']]

print(f"3. Saving V5 Data... (Buy: {len(df[df['Target']==1])}, Sell: {len(df[df['Target']==2])})")
final_df.to_csv("ai_ready_data_v5.csv", index=False)
print("✅ Success! V5 Data Created (Fixed).")