import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical

dataset = load_digits()
image_shape = (8, 8, 1)
num_classes = 10

y = dataset.target
y = to_categorical(y, num_classes)
x = dataset.data
x = np.array([data.reshape(image_shape) for data in x])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

model  = Sequential([
    Conv2D(5, 
           kernel_size=3, 
           activation='relu', 
           strides=1,
           input_shape=image_shape,
           padding='same'),
    Conv2D(5, 
           kernel_size=2,
           strides=1,
           activation='relu',
            padding='same'),
    Flatten(),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8)

predicts = model.predict(x_test)
predict = np.argmax(predicts, axis=1)
actual = np.argmax(y_test, axis=1)
print(classification_report(actual, predict))