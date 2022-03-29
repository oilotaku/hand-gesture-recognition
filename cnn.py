import numpy as np
import cv2
import sys
import os
import glob
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
import math

fdata = glob.glob('./new_train_data/train/*')
n = len(fdata)

def INPUT_Data_all(img_tatol_path, img_D_tatol_path):
    data_path = glob.glob(img_tatol_path)
    data_D_path = glob.glob(img_D_tatol_path)
    i = 0
    img_data_tatol = glob.glob(img_tatol_path + '/*')
    d = len(img_data_tatol)
    train_image = np.empty((d, 48, 64, 3))
    train_D_image = np.empty((d, 48, 64))
    train_image_tetol = np.empty((d, 48, 64, 4))
    train_label = np.full(d, 0)

    for path in data_path:
        label = path[-1]
        img_data = glob.glob(path + '/*')
        for file in img_data:
            img = cv2.imread(file)
            img = skinmask(img)
            img = findCon(img, img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
            # cv2.imshow('img',img)
            # cv2.waitKey()
            train_image[i] = img
            train_label[i] = label
            i += 1

    i = 0
    for path in data_D_path:
        img_data = glob.glob(path + '/*')
        for file in img_data:
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
            train_D_image[i] = img
            i += 1

    train_image = train_image.T
    train_D_image = train_D_image.T
    train_image_tetol = train_image_tetol.T
    train_image_tetol[0] = train_image[0]
    train_image_tetol[1] = train_image[1]
    train_image_tetol[2] = train_image[2]
    train_image_tetol[3] = train_D_image
    train_image_tetol = train_image_tetol.T


    return train_image_tetol/255.0, train_label

def skinmask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 轉換至YCrCb空間
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (7, 7), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu處理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    ker = np.ones((5, 5), np.uint8)
    res = cv2.dilate(res, ker, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # 腐蝕影像
    res = cv2.erode(res, kernel)

    return res

def findCon(img, rimg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    con, hie = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(con)):
        n, h, w = np.shape(con[j])
        if n >= 300:
            M = cv2.moments(con[j])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(rimg, con, -1, (255, 255, 255), 1)
            # cv2.circle(rimg, (cx, cy), 7, (255, 255, 255), -1)
            for i in range(n):
                if i % 60 == 0:
                    arr = con[j]
                    cv2.line(rimg, (cx, cy), arr[i][0], (255, 255, 255), 1)
                    continue
    return rimg

class TRAIN():
    (train_image, train_label) = INPUT_Data_all('./new_train_data/train/*', './new_train_data/train_D/*')
    # (test_image, test_label) = INPUT_Data_all('./train_data/RGB_1pose_3D/train/*', './train_data/RGB_1pose_3D/train_D/*')
    train_image = tf.keras.utils.normalize(train_image)
    # test_image = tf.keras.utils.normalize(test_image)

    per = np.random.permutation(train_label.shape[0])
    train_image = train_image[per, :, :]
    train_label = train_label[per]

    # per = np.random.permutation(test_label.shape[0])
    # test_image = test_image[per, :, :]
    # test_label = test_label[per]

    model = models.Sequential()

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(48, 64, 4)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2048, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(n, activation='softmax'))
    model.summary()

    #model.add(tf.keras.applications.resnet.ResNet101(include_top=True, weights=None, input_tensor=None,input_shape=(64, 48, 4), pooling=max, classes=1000))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    # history = model.fit(test_image, test_label, epochs=50, validation_split=0.2)
    history = model.fit(train_image, train_label, epochs=50, validation_split=0.5, batch_size=50)
    # history = model.fit(train_image, train_label, epochs=50, validation_data=(test_image, test_label))

    plt.plot(history.history['accuracy'], label='cnn_accuracy')
    plt.plot(history.history['val_accuracy'], label='cnn_val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    model.save('cnn_model_new.h5')
