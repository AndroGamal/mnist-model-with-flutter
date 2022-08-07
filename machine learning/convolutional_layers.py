import tensorflow as tf
from tensorflow import keras
#save model in file as name.tflite
def save_as_tflite(model,name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations =[tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types=[tf.float16]
    tflite_model = converter.convert()
    open(name + '.tflite', 'wb').write(tflite_model)
    print("End")

class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs={}):
        if(logs.get('accuracy')>.80):
            self.model.stop_training=True
end =callback()
mnist=keras.datasets.fashion_mnist
(train,label_train),(test,label_test)=mnist.load_data()
print()
print(test[0])
train=train/255.0
test=test/255.0
model=keras.Sequential([keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,tf.nn.relu),
    keras.layers.Dense(10,tf.nn.softmax)])
print(model.summary())

model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')
model.fit(train,label_train,epochs=1,callbacks=end)
# l=test[0].reshape(1,28,28,1)
# y=model.predict([l])

# # cv2.imwrite("image.png",l)
# # print(y[0])
# save_as_tflite(model=model,name="convolution")