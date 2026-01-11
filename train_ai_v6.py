import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import joblib

print("1. Loading Data (V5)...")
# We use the same V5 data because it has good features
df = pd.read_csv('ai_ready_data_v5.csv')

X = df.drop(columns=['Target']).values
y = df['Target'].values

print("2. Scaling Data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_v6.pkl')

# --- LSTM DATA RESHAPING ---
# LSTM needs 3D input: (Samples, Time Steps, Features)
# We are treating each row as 1 time step with 8 features
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

print("3. Building LSTM Brain (The Pro Model)...")
model = Sequential()

# LSTM Layer 1
# return_sequences=True because we have another LSTM layer after this
model.add(LSTM(128, return_sequences=True, input_shape=(1, X_scaled.shape[1])))
model.add(Dropout(0.3))

# LSTM Layer 2
model.add(LSTM(64))
model.add(Dropout(0.3))

# Dense Layers for Decision Making
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())

# Output Layer (3 Classes: Wait, Buy, Sell)
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("4. Training LSTM Started...")
# LSTM takes longer to train, so we give it 50 epochs
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

model.save('my_ai_model_v6.h5')
print("✅ Success! LSTM Model V6 Saved.")