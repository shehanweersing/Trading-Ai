# Trading-Ai 
# 🤖 AI-Powered XAUUSD Trading Bot (LSTM)

An advanced automated trading bot for **Gold (XAUUSD)** that uses **Deep Learning (LSTM)** to predict market movements. This bot analyzes historical price data, technical indicators (RSI, EMA, ATR), and market volatility to execute Buy/Sell trades automatically on **MetaTrader 5 (MT5)**.

## 🚀 Features

* **Deep Learning Brain:** Uses Long Short-Term Memory (LSTM) networks to understand market sequences and history.
* **Multi-Factor Analysis:**
    * **Heikin-Ashi Candles:** For smoother trend detection.
    * **RSI & EMA:** To identify overbought/oversold conditions and trends.
    * **ATR (Average True Range):** To measure volatility.
    * **Time-Aware:** Adjusts strategy based on the time of day (Session analysis).
* **Memory Features:** Uses Lag Features to "remember" past candle data for better accuracy.
* **Live Auto-Trading:** Connects directly to MT5 terminal to execute trades in real-time.
* **Risk Management:** Includes logic for "Wait" signals during choppy markets.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** TensorFlow / Keras (LSTM)
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Trading Platform:** MetaTrader 5 (MT5) Integration

## 📂 Project Structure

* `get_data.py` - Fetches historical raw data (OHLC) from MT5.
* `prepare_data_v6.py` - Calculates indicators, lag features, and cleans data for the AI.
* `train_ai_v6.py` - Trains the LSTM Neural Network and saves the model (`.h5`) and scaler (`.pkl`).
* `trade_live_v6.py` - The main bot script that runs live, predicts signals, and executes trades.

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/ai-trading-bot.git](https://github.com/yourusername/ai-trading-bot.git)
    cd ai-trading-bot
    ```

2.  **Install Requirements**
    You need the following Python libraries:
    ```bash
    pip install MetaTrader5 pandas numpy tensorflow scikit-learn
    ```

3.  **MetaTrader 5 Setup**
    * Open your MT5 Terminal.
    * Go to **Tools > Options > Expert Advisors**.
    * Enable **"Allow automated trading"**.

## 🚀 How to Run

Follow these steps in order:

**Step 1: Fetch Data**
Open MT5 and ensure `XAUUSD` is visible in Market Watch. Then run:
```bash
python get_data.py

Step 2: Prepare Data This calculates indicators and creates the 3D dataset for LSTM.

Bash

python prepare_data_v6.py
Step 3: Train the AI This will train the model. It may take a few minutes.

Bash

python train_ai_v6.py
Output: my_ai_model_v6.h5 and scaler_v6.pkl

Step 4: Start Live Trading Run the bot to start analyzing the market in real-time.

Bash

python trade_live_v6.py
