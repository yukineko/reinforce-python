import numpy as np
#from tensorflow.python import keras as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid

model = Sequential([
    Dense(units=4, input_shape=((2,)))
])

weight, bias = model.layers[0].get_weights()
print("weight shape is {}".format(weight.shape))
print("Bias shape is {}".format(bias.shape))

x = np.random.rand(1,2)
y = model.predict(x)
print("x is {} y is {}".format(x.shape, y.shape))