import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %matplotlib inline

from tqdm import tqdm_notebook, tnrange
# from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# Set some parameters

im_width = 128
im_height = 128
border = 5


ids = next(os.walk("images"))[2] # list of names all images in the given path
print("No. of images = ", len(ids))
# No. of images =  4000

X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

# Load the images and masks into arrays

# tqdm is used to display the progress bar
for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
    # Load images
    img = load_img("images/"+id_, grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # Load masks
    mask = img_to_array(load_img("masks/"+id_, grayscale=True))
    mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)
    # Save images
    X[n] = x_img/255.0
    y[n] = mask/255.0


# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


# Below code can be used to visualize the images and corresponding masks

# Visualize any randome image along with the mask
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0 # salt indicator

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 15))

ax1.imshow(X_train[ix, ..., 0], cmap = 'seismic', interpolation = 'bilinear')
if has_mask: # if salt
    # draw a boundary(contour) in the original image separating salt and non-salt areas
    ax1.contour(y_train[ix].squeeze(), colors = 'k', linewidths = 5, levels = [0.5])
ax1.set_title('Seismic')

ax2.imshow(y_train[ix].squeeze(), cmap = 'gray', interpolation = 'bilinear')
ax2.set_title('Salt')


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


model.summary()

# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# img (InputLayer)                (None, 128, 128, 1)  0
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 128, 128, 16) 160         img[0][0]
# __________________________________________________________________________________________________
# batch_normalization_2 (BatchNor (None, 128, 128, 16) 64          conv2d_2[0][0]
# __________________________________________________________________________________________________
# activation_2 (Activation)       (None, 128, 128, 16) 0           batch_normalization_2[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 16)   0           activation_2[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 64, 64, 16)   0           max_pooling2d_1[0][0]
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 64, 64, 32)   4640        dropout_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_4 (BatchNor (None, 64, 64, 32)   128         conv2d_4[0][0]
# __________________________________________________________________________________________________
# activation_4 (Activation)       (None, 64, 64, 32)   0           batch_normalization_4[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 32)   0           activation_4[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 32, 32, 32)   0           max_pooling2d_2[0][0]
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 32, 32, 64)   18496       dropout_2[0][0]
# __________________________________________________________________________________________________
# batch_normalization_6 (BatchNor (None, 32, 32, 64)   256         conv2d_6[0][0]
# __________________________________________________________________________________________________
# activation_6 (Activation)       (None, 32, 32, 64)   0           batch_normalization_6[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 64)   0           activation_6[0][0]
# __________________________________________________________________________________________________
# dropout_3 (Dropout)             (None, 16, 16, 64)   0           max_pooling2d_3[0][0]
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 16, 16, 128)  73856       dropout_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_8 (BatchNor (None, 16, 16, 128)  512         conv2d_8[0][0]
# __________________________________________________________________________________________________
# activation_8 (Activation)       (None, 16, 16, 128)  0           batch_normalization_8[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 128)    0           activation_8[0][0]
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 8, 8, 128)    0           max_pooling2d_4[0][0]
# __________________________________________________________________________________________________
# conv2d_10 (Conv2D)              (None, 8, 8, 256)    295168      dropout_4[0][0]
# __________________________________________________________________________________________________
# batch_normalization_10 (BatchNo (None, 8, 8, 256)    1024        conv2d_10[0][0]
# __________________________________________________________________________________________________
# activation_10 (Activation)      (None, 8, 8, 256)    0           batch_normalization_10[0][0]
# __________________________________________________________________________________________________
# conv2d_transpose_1 (Conv2DTrans (None, 16, 16, 128)  295040      activation_10[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 16, 16, 256)  0           conv2d_transpose_1[0][0]
#                                                                  activation_8[0][0]
# __________________________________________________________________________________________________
# dropout_5 (Dropout)             (None, 16, 16, 256)  0           concatenate_1[0][0]
# __________________________________________________________________________________________________
# conv2d_12 (Conv2D)              (None, 16, 16, 128)  295040      dropout_5[0][0]
# __________________________________________________________________________________________________
# batch_normalization_12 (BatchNo (None, 16, 16, 128)  512         conv2d_12[0][0]
# __________________________________________________________________________________________________
# activation_12 (Activation)      (None, 16, 16, 128)  0           batch_normalization_12[0][0]
# __________________________________________________________________________________________________
# conv2d_transpose_2 (Conv2DTrans (None, 32, 32, 64)   73792       activation_12[0][0]
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 32, 32, 128)  0           conv2d_transpose_2[0][0]
#                                                                  activation_6[0][0]
# __________________________________________________________________________________________________
# dropout_6 (Dropout)             (None, 32, 32, 128)  0           concatenate_2[0][0]
# __________________________________________________________________________________________________
# conv2d_14 (Conv2D)              (None, 32, 32, 64)   73792       dropout_6[0][0]
# __________________________________________________________________________________________________
# batch_normalization_14 (BatchNo (None, 32, 32, 64)   256         conv2d_14[0][0]
# __________________________________________________________________________________________________
# activation_14 (Activation)      (None, 32, 32, 64)   0           batch_normalization_14[0][0]
# __________________________________________________________________________________________________
# conv2d_transpose_3 (Conv2DTrans (None, 64, 64, 32)   18464       activation_14[0][0]
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 64, 64, 64)   0           conv2d_transpose_3[0][0]
#                                                                  activation_4[0][0]
# __________________________________________________________________________________________________
# dropout_7 (Dropout)             (None, 64, 64, 64)   0           concatenate_3[0][0]
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (None, 64, 64, 32)   18464       dropout_7[0][0]
# __________________________________________________________________________________________________
# batch_normalization_16 (BatchNo (None, 64, 64, 32)   128         conv2d_16[0][0]
# __________________________________________________________________________________________________
# activation_16 (Activation)      (None, 64, 64, 32)   0           batch_normalization_16[0][0]
# __________________________________________________________________________________________________
# conv2d_transpose_4 (Conv2DTrans (None, 128, 128, 16) 4624        activation_16[0][0]
# __________________________________________________________________________________________________
# concatenate_4 (Concatenate)     (None, 128, 128, 32) 0           conv2d_transpose_4[0][0]
#                                                                  activation_2[0][0]
# __________________________________________________________________________________________________
# dropout_8 (Dropout)             (None, 128, 128, 32) 0           concatenate_4[0][0]
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (None, 128, 128, 16) 4624        dropout_8[0][0]
# __________________________________________________________________________________________________
# batch_normalization_18 (BatchNo (None, 128, 128, 16) 64          conv2d_18[0][0]
# __________________________________________________________________________________________________
# activation_18 (Activation)      (None, 128, 128, 16) 0           batch_normalization_18[0][0]
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (None, 128, 128, 1)  17          activation_18[0][0]
# ==================================================================================================
# Total params: 1,179,121
# Trainable params: 1,177,649
# Non-trainable params: 1,472


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


results = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=callbacks,\
                    validation_data=(X_valid, y_valid))



# Train on 3600 samples, validate on 400 samples
# # Epoch 1/50
# # 3600/3600 [==============================] - 15s 4ms/step - loss: 0.4410 - acc: 0.8041 - val_loss: 1.4167 - val_acc: 0.5940
# #
# # Epoch 00001: val_loss improved from inf to 1.41666, saving model to model-tgs-salt.h5
# # Epoch 2/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.3112 - acc: 0.8726 - val_loss: 0.4795 - val_acc: 0.8208
# #
# # Epoch 00002: val_loss improved from 1.41666 to 0.47946, saving model to model-tgs-salt.h5
# # Epoch 3/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2882 - acc: 0.8794 - val_loss: 0.3361 - val_acc: 0.8698
# #
# # Epoch 00003: val_loss improved from 0.47946 to 0.33606, saving model to model-tgs-salt.h5
# # Epoch 4/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2688 - acc: 0.8894 - val_loss: 0.3677 - val_acc: 0.8521
# #
# # Epoch 00004: val_loss did not improve from 0.33606
# # Epoch 5/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2508 - acc: 0.8943 - val_loss: 0.3899 - val_acc: 0.8385
# #
# # Epoch 00005: val_loss did not improve from 0.33606
# # Epoch 6/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2339 - acc: 0.9000 - val_loss: 0.3212 - val_acc: 0.8736
# #
# # Epoch 00006: val_loss improved from 0.33606 to 0.32124, saving model to model-tgs-salt.h5
# # Epoch 7/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2272 - acc: 0.9017 - val_loss: 0.3125 - val_acc: 0.8857
# #
# # Epoch 00007: val_loss improved from 0.32124 to 0.31247, saving model to model-tgs-salt.h5
# # Epoch 8/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2220 - acc: 0.9033 - val_loss: 0.3552 - val_acc: 0.8359
# #
# # Epoch 00008: val_loss did not improve from 0.31247
# # Epoch 9/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2079 - acc: 0.9093 - val_loss: 0.3468 - val_acc: 0.8842
# #
# # Epoch 00009: val_loss did not improve from 0.31247
# # Epoch 10/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.2059 - acc: 0.9107 - val_loss: 0.2791 - val_acc: 0.8726
# #
# # Epoch 00010: val_loss improved from 0.31247 to 0.27907, saving model to model-tgs-salt.h5
# # Epoch 11/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1985 - acc: 0.9138 - val_loss: 0.2447 - val_acc: 0.8952
# #
# # Epoch 00011: val_loss improved from 0.27907 to 0.24470, saving model to model-tgs-salt.h5
# # Epoch 12/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1933 - acc: 0.9146 - val_loss: 0.2716 - val_acc: 0.8972
# #
# # Epoch 00012: val_loss did not improve from 0.24470
# # Epoch 13/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1857 - acc: 0.9185 - val_loss: 0.2410 - val_acc: 0.8911
# #
# # Epoch 00013: val_loss improved from 0.24470 to 0.24098, saving model to model-tgs-salt.h5
# # Epoch 14/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1825 - acc: 0.9192 - val_loss: 0.2371 - val_acc: 0.8988
# #
# # Epoch 00014: val_loss improved from 0.24098 to 0.23709, saving model to model-tgs-salt.h5
# # Epoch 15/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1752 - acc: 0.9229 - val_loss: 0.2827 - val_acc: 0.9023
# #
# # Epoch 00015: val_loss did not improve from 0.23709
# # Epoch 16/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1742 - acc: 0.9241 - val_loss: 0.2410 - val_acc: 0.9049
# #
# # Epoch 00016: val_loss did not improve from 0.23709
# # Epoch 17/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1635 - acc: 0.9271 - val_loss: 0.2534 - val_acc: 0.8914
# #
# # Epoch 00017: val_loss did not improve from 0.23709
# # Epoch 18/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1706 - acc: 0.9241 - val_loss: 0.3032 - val_acc: 0.8860
# #
# # Epoch 00018: val_loss did not improve from 0.23709
# # Epoch 19/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1579 - acc: 0.9280 - val_loss: 0.2299 - val_acc: 0.9087
# #
# # Epoch 00019: val_loss improved from 0.23709 to 0.22993, saving model to model-tgs-salt.h5
# # Epoch 20/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1588 - acc: 0.9285 - val_loss: 0.2512 - val_acc: 0.8970
# #
# # Epoch 00020: val_loss did not improve from 0.22993
# # Epoch 21/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1496 - acc: 0.9317 - val_loss: 0.2093 - val_acc: 0.9073
# #
# # Epoch 00021: val_loss improved from 0.22993 to 0.20935, saving model to model-tgs-salt.h5
# # Epoch 22/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1445 - acc: 0.9336 - val_loss: 0.2978 - val_acc: 0.8607
# #
# # Epoch 00022: val_loss did not improve from 0.20935
# # Epoch 23/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1448 - acc: 0.9328 - val_loss: 0.2253 - val_acc: 0.9056
# #
# # Epoch 00023: val_loss did not improve from 0.20935
# # Epoch 24/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1361 - acc: 0.9371 - val_loss: 0.2306 - val_acc: 0.9063
# #
# # Epoch 00024: val_loss did not improve from 0.20935
# # Epoch 25/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1285 - acc: 0.9403 - val_loss: 0.3140 - val_acc: 0.8888
# #
# # Epoch 00025: val_loss did not improve from 0.20935
# # Epoch 26/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1272 - acc: 0.9403 - val_loss: 0.2328 - val_acc: 0.9139
# #
# # Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
# #
# # Epoch 00026: val_loss did not improve from 0.20935
# # Epoch 27/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.1038 - acc: 0.9505 - val_loss: 0.1870 - val_acc: 0.9190
# #
# # Epoch 00027: val_loss improved from 0.20935 to 0.18702, saving model to model-tgs-salt.h5
# # Epoch 28/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0942 - acc: 0.9540 - val_loss: 0.1918 - val_acc: 0.9184
# #
# # Epoch 00028: val_loss did not improve from 0.18702
# # Epoch 29/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0920 - acc: 0.9553 - val_loss: 0.1993 - val_acc: 0.9195
# #
# # Epoch 00029: val_loss did not improve from 0.18702
# # Epoch 30/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0919 - acc: 0.9551 - val_loss: 0.1944 - val_acc: 0.9180
# #
# # Epoch 00030: val_loss did not improve from 0.18702
# # Epoch 31/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0849 - acc: 0.9588 - val_loss: 0.2012 - val_acc: 0.9186
# #
# # Epoch 00031: val_loss did not improve from 0.18702
# # Epoch 32/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0849 - acc: 0.9582 - val_loss: 0.2090 - val_acc: 0.9155
# #
# # Epoch 00032: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
# #
# # Epoch 00032: val_loss did not improve from 0.18702
# # Epoch 33/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0822 - acc: 0.9592 - val_loss: 0.2035 - val_acc: 0.9178
# #
# # Epoch 00033: val_loss did not improve from 0.18702
# # Epoch 34/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0797 - acc: 0.9607 - val_loss: 0.2032 - val_acc: 0.9182
# #
# # Epoch 00034: val_loss did not improve from 0.18702
# # Epoch 35/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0823 - acc: 0.9598 - val_loss: 0.2044 - val_acc: 0.9179
# #
# # Epoch 00035: val_loss did not improve from 0.18702
# # Epoch 36/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0809 - acc: 0.9601 - val_loss: 0.2042 - val_acc: 0.9178
# #
# # Epoch 00036: val_loss did not improve from 0.18702
# # Epoch 37/50
# # 3600/3600 [==============================] - 10s 3ms/step - loss: 0.0807 - acc: 0.9604 - val_loss: 0.2035 - val_acc: 0.9179
# #
# # Epoch 00037: ReduceLROnPlateau reducing learning rate to 1e-05.
# #
# # Epoch 00037: val_loss did not improve from 0.18702
# # Epoch 00037: early stopping


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()


#Inference - conclusion


# load the best model
model.load_weights('model-tgs-salt.h5')

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)
#
# 400/400 [==============================] - 0s 1ms/step
# Out[20]:
# [0.18702135920524599, 0.9190425109863282]

# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

#
# 3600/3600 [==============================] - 5s 1ms/step
# 400/400 [==============================] - 0s 848us/step


# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Predicted binary');

# Predictions on training set

# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)

plot_sample(X_valid, y_valid, preds_val, preds_val_t)  # For some times.




