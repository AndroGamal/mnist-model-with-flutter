import tensorflow as tf
from tensorflow import keras
import numpy as np

model=keras.Sequential(keras.layers.Dense(units=1,input_shape=[1]))
model.compile(optimizer='sgd',loss='mean_squared_error')
xs=np.array([1,2,3,4,5,6,7,8,9,10,11,12],dtype=float)
ys=np.array([2,4,6,8,10,12,14,16,18,20,22,24],dtype=float)

model.fit(xs,ys,epochs=1000)
y=model.predict([20.0])
print(y)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("End")