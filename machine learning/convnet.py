import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

os.chdir("./")


# to stop train class
class callback(tf.keras.callbacks.Callback):
    old=0
    # def on_epoch_begin(self, epoch,logs={}):
    #     # self.lr=self.model.optimizer.learning_rate = 0.1
    #     self.model.optimizer.optimizer_specs[0]["optimizer"].lr=0.1
    #     print(self.model.optimizer.optimizer_specs[0]["optimizer"].lr)
    #     for optimizer in self.model.optimizer.optimizer_specs:
    #         optimizer["optimizer"].lr=self.learn(epoch)

    def on_epoch_end(self, epoch,logs={}):
        # os.system('clear')
        val_accuracy=logs['val_T']/(logs['val_T']+logs['val_F'])
        accuracy=logs['T']/(logs['T']+logs['F'])
        print("\n - lr accuracy: ",round(accuracy*100,3),"%")
        print(" - lr val_accuracy: ",round(val_accuracy*100,3),"%")
        tf.print(" - lr Adam: ",round(float(self.model.optimizer.optimizer_specs[0]["optimizer"].lr),5))
        tf.print(" - lr RMSprop: ",round(float(self.model.optimizer.optimizer_specs[1]["optimizer"].lr),5))
        tf.print(" - lr SGD: ",round(float(self.model.optimizer.optimizer_specs[2]["optimizer"].lr),8))
        if(self.old>=val_accuracy):
            self.model.optimizer.optimizer_specs[0]["optimizer"].lr=0.005/(epoch)
            self.model.optimizer.optimizer_specs[1]["optimizer"].lr=self.model.optimizer.optimizer_specs[1]["optimizer"].lr-0.0001
            self.model.optimizer.optimizer_specs[2]["optimizer"].lr=1e-09 * 1000 ** ((1000-epoch)/1000)
        if (logs['val_acc']>0.90  or accuracy>=0.90) and logs['acc']>=0.90:
            self.model.stop_training=True
        self.old=val_accuracy

# genrate image to train s
def genrate_image():
    tran_data=ImageDataGenerator(rescale=1./225,shear_range=.20,zoom_range=0.20,brightness_range=(.50,1.50),rotation_range=45)
    # tran_data=ImageDataGenerator(rescale=1./225)
    train_image_genrater=tran_data.flow_from_directory(directory="image/train",target_size=(300,300),batch_size=20,
                                                       classes={'cats': 0,"dogs":1,'horses': 2, 'humans':3 },shuffle=True)
    tran_data=ImageDataGenerator()
    val_image_genrater=tran_data.flow_from_directory(directory="image/validation",target_size=(300,300),batch_size=20,
                                                     classes={'cats': 0,"dogs":1,'horses': 2, 'humans':3 },shuffle=True)
    # print(train_image_genrater.class_indices)
    # exit(0)
    return train_image_genrater,val_image_genrater

end =callback()
train,validation=genrate_image()

# def representative_dataset():
#     for t in train:
#       yield [t[0].astype(tf.float16)]


#save model in file as name.tflite
def save_as_tflite(model,name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations =[tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types=[tf.float16]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                                    tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(name + '.tflite', 'wb').write(tflite_model)
    print("End")




# def p(outs):
#     max=[]
#     out=[]
#     o=[]
#     for i in range(0,3):
#         max.append(outs[i][0])
#         for r in range(0,3):
#             max[i]=tf.cond(max[i]<outs[i][r],lambda:outs[i][r],lambda:max[i])
#     for i in range(0,3):
#         for r in range(0,3):
#             o.append(tf.cond(max[i]==outs[i][r],lambda:1.0,lambda:0.0))
#         out.append(o)
#         o=[]
#     r=tf.convert_to_tensor(out,tf.float32)
#     r=tf.reshape(r,(-1,-1))
#     r=tf.reshape(r,(-1,3))
#     return r





model=keras.Sequential([keras.layers.Conv2D(4,(3,3),activation=tf.nn.crelu,input_shape=(300,300,3)),
    keras.layers.MaxPooling2D(3,3),
    keras.layers.Conv2D(8,(2,2),activation=tf.nn.crelu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(16,(3,3),activation=tf.nn.crelu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation=tf.nn.crelu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation=tf.nn.crelu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation=tf.nn.crelu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(16,tf.nn.crelu),
    keras.layers.Dense(4,tf.nn.softmax),])
    # keras.layers.Lambda(p)])



print(model.summary())
tf.random.set_seed(80)


model.compile(optimizer=tfa.optimizers.MultiOptimizer([ (keras.optimizers.Adamax(lr=0.005),model.layers[:5]),
                                                        (keras.optimizers.RMSprop(lr=0.005),model.layers[5:10]),
                                                        (keras.optimizers.SGD(lr=1e-6),model.layers[10:])])
              ,metrics=["acc",keras.metrics.FalseNegatives(name="F"),keras.metrics.TruePositives(name="T")]
              ,loss=['mean_squared_error',"binary_crossentropy",keras.losses.Huber()])
# model.compile(optimizer=keras.optimizers.Adam(),sample_weight_mode=0,loss_weights=[100,0.50,0.001],metrics=['accuracy'],loss=[tf.keras.losses.Huber(),'binary_crossentropy','mean_squared_error'])
# model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),metrics=['accuracy'],loss='binary_crossentropy')
# model.compile(optimizer=keras.optimizers.SGD(),sample_weight_mode=0,loss_weights=[100,0.50,0.001],metrics=['accuracy'],loss=[tf.keras.losses.Huber(),'binary_crossentropy','mean_squared_error'])
# 'sparse_categorical_crossentropy'
# 'poisson'
# tf.keras.metrics.FalseNegatives()
# 'binary_crossentropy'
# 'mean_squared_error'
# 'mean_absolute_error'
# tf.keras.losses.Huber()
model.fit(train,validation_data=validation,
          epochs=150,batch_size=20,callbacks=[end])
save_as_tflite(model,"human_or_horse")