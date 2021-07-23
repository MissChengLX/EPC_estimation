# -*- coding: UTF-8 -*-
import keras
#import theano

#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
from keras import backend as K
from keras import Model, Input
from keras.applications import VGG16, ResNet50

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import load_model
from keras.preprocessing import image
from PIL import ImageFile
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)
# 只使用第三块GPU。
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import TensorBoard
#tbCallBack = TensorBoard(log_dir="./model", histogram_freq=1,write_grads=True)
#history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=0.2,callbacks=[tbCallBack])

#VGG16_notop_path='./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
# ImageFile.LOAD_TRUNCATED_IMAGES = True
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7  #限制GPU内存占用率
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



EPOCHS = 30
BATCH_SIZE = 32
#DATA_PATH = r"D:\2013"
#DATA_PATH = r"I:\Wuhan_google\class"
DATA_PATH = r'H:\Evolution_Dataset\Shenzhen\class'
restnet50 = './models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# -------------准备数据--------------------------
datagen = ImageDataGenerator()
# load and iterate training dataset   shuffle=True 判断是否打乱
train_it = datagen.flow_from_directory(DATA_PATH + '/train/', target_size=(100, 100),
                                           shuffle = False,
                                           class_mode='categorical',
                                           batch_size=BATCH_SIZE)
# load and iterate validation dataset
val_it = datagen.flow_from_directory(DATA_PATH + '/validation/', target_size=(100, 100),
                                         shuffle = False,
                                         class_mode='categorical',
                                         batch_size=BATCH_SIZE)
# load and iterate test dataset
test_it = datagen.flow_from_directory(DATA_PATH + '/test/', target_size=(100, 100),
                                          shuffle = False,
                                          class_mode='categorical',
                                          batch_size=BATCH_SIZE)

def Train():
    nb_classes =3
    # -------------加载VGG模型并且添加自己的层----------------------
    # 这里自己添加的层需要不断调整超参数来提升结果，输出类别更改softmax层即可
    #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    # 参数说明：inlucde_top:是否包含最上方的Dense层，input_shape：输入的图像大小(width,height,channel)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    # base_model.summary()
    # x = base_model.output
    # x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1, activation='sigmoid')(x)
    # predictions = Dense(nb_classes, activation='softmax')(x)
    # #x = Dense(nb_classes, activation='sigmoid')(x)
    # model = Model(inputs=base_model.inputs, outputs=predictions)
   # model = multi_gpu_model(model, 2)  # GPU个数为2
    # -----------控制需要FineTune的层数，不FineTune的就直接冻结
    #---冻结前5层------
    # for layer in base_model.layers[:5]:
    #     layer.trainable = False

    for layers in base_model.layers[:]:
        layers.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # ----------编译，设置优化器，损失函数，性能指标
    sgd = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # ----------设置tensorboard,用来观察acc和loss的曲线---------------
    # tbCallBack = TensorBoard(log_dir='./logs/' + TIMESTAMP,  # log 目录
    # histogram_freq = 0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    # batch_size = 16,  # 用多大量的数据计算直方图
    # write_graph = True,  # 是否存储网络结构图
    # write_grads = True,  # 是否可视化梯度直方图
    # write_images = True,  # 是否可视化参数
    # embeddings_freq = 0,
    # embeddings_layer_names = None,
    # embeddings_metadata = None)

# ---------设置自动保存点，acc最好的时候就会自动保存一次，会覆盖之前的存档---------------
    checkpoint = ModelCheckpoint(filepath='HatNewModel.h5', monitor='acc', mode='auto', save_best_only='True')

# ----------开始训练---------------------------------------------

#     history = model.fit_generator(train_it,steps_per_epoch=BATCH_SIZE,
#                                   epochs=EPOCHS, verbose=1,
#                                   #callbacks=[tbCallBack, checkpoint],
#                                   validation_data=test_it, validation_steps=BATCH_SIZE)
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping

    early_stop = EarlyStopping(monitor='val_loss', patience=13)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', factor=0.2)
    callbacks = [early_stop, reduce_lr]

    history = model.fit_generator(train_it, steps_per_epoch=BATCH_SIZE,#train_it.samples // BATCH_SIZE,
        epochs=100,
        validation_data=val_it,
        validation_steps=BATCH_SIZE #val_it.samples // BATCH_SIZE,
        #callbacks=callbacks
                                  )
    # evaluate model
    loss = model.evaluate_generator(test_it, steps=24)
    return history


# ---------------画图，将训练时的acc和loss都绘制到图上---------------------
def plot_training(history):
    plt.figure(12)
    plt.subplot(121)
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label='train_accuracy')
    plt.plot(epochs, val_acc, 'r', label='test_accuracy')
    plt.title('Train and Test accuracy')
    plt.legend()

    plt.subplot(122)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='test_loss')
    plt.title('Train and Test loss')
    plt.legend()

    plt.show()

# -------------预测单个图像--------------------------------------
def Predict(imgPath):
    model = load_model("SAVE_MODEL_NAME")
    #img = image.load_img(imgPath, target_size=(100, 100))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    res = model.predict_gennerater(test_it)
    #res = model.predict(x)
    print(np.argmax(res, axis=1)[0])

if __name__ == '__main__':
    history=Train()
    plot_training(history)