import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
model = Sequential([
    Dense(units=4, input_shape=((2,)), activation=sigmoid),
    Dense(units=4)
])


batch = np.random.rand(3,2)
y = model.predict(batch)
print("batch is {} y is {}".format(batch.shape, y.shape))