from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

# データ読み込み
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル構築
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # 明示的なInput 最近はこれを書く
    Dense(64, activation='softmax'),
    Dense(64, activation='relu'),
    Dense(1)  # 回帰なので活性化なし
])

# コンパイル
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 学習
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 評価
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')

predicts = model.predict(X_test)
result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': np.reshape(predicts, (-1,))
})
limit = np.max(y_test)
result.plot.scatter(x='Actual', y='Predicted', xlim=(0, limit), ylim=(0, limit))
plt.show()