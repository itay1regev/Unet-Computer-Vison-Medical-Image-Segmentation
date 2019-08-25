from __future__ import print_function
############### Imports ######################################################################################
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgba2rgb
# from scipy.misc import imshow
from keras.models import Model
# from tensorflow.python.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, \
#    BatchNormalization, Activation
# from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.optimizers import SGD
#from tensorflow.python.keras.utils import get_custom_objects
#import glob
#from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
#from tensorflow.python.keras.models import load_model


# from itertools import chain



from imgaug import augmenters as imt

from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob
from tqdm import tqdm_notebook, tnrange
from skimage.io import imread, imshow, concatenate_images
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import get_custom_objects
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Dropout, BatchNormalization, Activation
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


##############################################################################################################


############### TRAINING DATA CONSTANTS ######################################################################

weight_dir = 'Daniel_Shriki_Itay_Regev_model.h5'
path_to_data_dir = "/media/dani94/5203b9a4-e9d7-4808-b433-8f94413806ca/project1/class3_data"
image_name = "img.png"
mask_name = "cumulativeMask.png"

image_height_for_training = 256
image_width_for_training = 256


##############################################################################################################


class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)


def augment_images_masks(batch_image, batch_mask):
    aug = imt.SomeOf(2, [
        imt.Fliplr(0.5),
        imt.Flipud(0.5),
        imt.Affine(rotate=45)
    ])
    aug = aug.to_deterministic()
    new_img_batch = aug.augment_images(batch_image)
    new_mask_batch = aug.augment_images(batch_mask)
    return new_img_batch, new_mask_batch


def augment_images_only(batch_image):
    aug = imt.OneOf([
        imt.Sharpen(alpha=0.3),
        # imt.AdditiveGaussianNoise(scale=0.1*255),
        imt.GaussianBlur(sigma=(0, 0.2))
    ])

    aug = aug.to_deterministic()
    new_img_batch = aug.augment_images(batch_image)
    return new_img_batch


############### READING THE DATA #############################################################################
def get_training_data(path_to_data_dir):
    """
    Find training and validation images in the given directory.
    :param path_to_data_dir: the absolute path to a data directory.
    :return:
            train_file_names: a list of training file names.
            val_file_names: a list of validation file names.
    """
    train_path = os.path.join(path_to_data_dir, "train")
    val_path = os.path.join(path_to_data_dir, "val")

    print("Trying to find training and validation data in folders:", train_path, val_path)

    train_file_names = glob.glob(train_path + "/*/*/" + image_name)
    val_file_names = glob.glob(val_path + "/*/*/" + image_name)

    # If this assert is reached, path_to_data_dir is probably wrong
    assert (len(train_file_names) > 0 and len(val_file_names) > 0)

    print("Found", len(train_file_names), "training files")
    print("Found", len(val_file_names), "validation files")

    return train_file_names, val_file_names


def load_image(path_to_image):
    """
    Loads png image from path_to_image to a numpy array.
    :param path_to_image: the absolute path to a png cell image.
    :return: a 3-dimensional numpy array of shape (height, width, 3) representing a RGB cell image.
    """
    img_array = imread(path_to_image)
    img_array_as_rgb = rgba2rgb(img_array).astype('float')

    return img_array_as_rgb


def preprocess_image(img_array):
    """
    Perform preprocessing of a cell image before feeding it to the model.
    :param img_array: a 3-dimensional numpy array of shape (height, width 3) representing a RGB cell image.
    :return: a processed version of the input image.
    """
    return img_array


def load_mask(path_to_mask):
    """
    Loads png binary mask from path_to_mask to a numpy array.
    :param path_to_mask: the absolute path to a png binary mask image.
    :return: a 2-dimensional numpy array of shape (height, width) representing a binary cell mask.
    """
    mask_array = imread(path_to_mask, as_gray=True)

    return mask_array


def preprocess_mask(mask_array):
    """
    Perform preprocessing of a binary mask before feeding it to the model.
    :param mask_array: a 2-dimensional numpy array of shape (height, width) representing a a binary cell mask.
    :return: a processed version of the input mask.
    """
    mask_array[mask_array > 0] = 1

    return mask_array


##############get_training_data################################################################################################


############### DATA GENERATOR ###############################################################################
def generate_training_batches(relevant_file_names, batch_size):
    # This function is called twice: once for training and once for validation, and generates data for the model

    num_files = len(relevant_file_names)

    while True:
        # One iteration of the outer while loop == one epoch of the model.
        # The order of the files is randomized per epoch by the following line:
        random_index_order = np.random.permutation(num_files)

        # Go over list of files, feeding batch_size files to the model at a time
        for batch_index in range(0, num_files - batch_size + 1, batch_size):
            for b in range(0, batch_size):
                cur_ind = random_index_order[batch_index + b]

                # Obtain paths to single example + corresponding segmentation mask
                image_path = relevant_file_names[cur_ind]
                mask_path = image_path.replace(image_name, mask_name)

                # Load example + mask as numpy arrays:
                img_array = load_image(image_path)
                img_array = preprocess_image(img_array)

                mask_array = load_mask(mask_path)
                mask_array = preprocess_mask(mask_array)

                # For debugging purposes, show image + mask after loading
                # imshow(img_array)
                # imshow(mask_array)

                # Add channel to mask array (as required by Tensorflow, 'channels_last')
                # Image itself already has 3 channels so no expansion is required
                mask_array = mask_array[..., np.newaxis]

                # Add dimensions to both image and mask so that they can be concatenated along this axis:
                img_array = np.expand_dims(img_array, axis=0)
                mask_array = np.expand_dims(mask_array, axis=0)

                if b == 0:
                    # First example in batch
                    batch_img = img_array
                    batch_mask = mask_array
                else:
                    batch_img = np.concatenate((batch_img, img_array), axis=0)
                    batch_mask = np.concatenate((batch_mask, mask_array), axis=0)

            # yield (batch_img, batch_mask)
            aug_batch_img, aug_batch_mask = augment_images_masks(batch_img, batch_mask)
            yield (aug_batch_img, aug_batch_mask)
            aug2_batch_img = augment_images_only(batch_img)
            yield (aug2_batch_img, batch_mask)
            batch_img = None
            batch_mask = None


##############################################################################################################


############### LEARNING RATE DECAY ##########################################################################

def lr_fun(epoch_num):
    if epoch_num < 10:
        lr = 5e-4
    elif epoch_num >= 10 and epoch_num < 25:
        lr = 1e-5
    elif epoch_num > 25 and epoch_num < 75:
        lr = 1e-6
    else:
        lr = 1e-6
    return lr


##############################################################################################################


############### METRICS AND TRAINING LOSS ####################################################################
def dice_coef(y_true, y_pred):
    # calculating the DICE coefficient with a smoothing term
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    smooth = 1

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    # calculating the DICE loss function for GD
    return 1 - dice_coef(y_true, y_pred)


#############################################################################################################


################## MODEL ARCHITECTURE DEFINITION ############################################################


###### FUNCTIONS DEFINING U-NET BLOCKS ######
def convolution_block(input_layer, num_filters, kernel_size, activation_func):

    conv1 = Conv2D(filters=num_filters, kernel_size=kernel_size,
                   kernel_initializer='he_uniform', padding='same')(input_layer)
    act_conv1 = Activation(activation_func)(conv1)
    conv2 = Conv2D(filters=num_filters, kernel_size=kernel_size,
                   kernel_initializer='he_uniform', padding='same')(act_conv1)
    act_conv2 = Activation(activation_func)(conv2)
    normal_layer = BatchNormalization()(act_conv2)
    return normal_layer


def conv_downsample_block(input_layer, num_filters, kernel_size, activation_func, max_pool_shape):
    pool = MaxPooling2D(pool_size=max_pool_shape)(input_layer)
    conv_block = convolution_block(pool, num_filters, kernel_size, activation_func)
    return conv_block


def conv_upsample_block(input_layer, skip_connection_layer, num_filters, kernel_size, activation_func,
                        upsampling_shape):
    upsampling = UpSampling2D(size=upsampling_shape)(input_layer)
    concatenate_skip = concatenate([upsampling, skip_connection_layer], axis=3)
    conv_block = convolution_block(concatenate_skip, num_filters, kernel_size, activation_func)
    dropout = Dropout(0.2)(conv_block)
    return dropout


###### U-NET ARCHITECTURE ######
def get_unet(img_rows, img_cols, first_layer_num_filters=32, num_classes=1):
    inputs = Input((img_rows, img_cols, 3))

    ###### ENCODER BRANCH ######
    # leaky_relu = LeakyReLU(alpha=0.2)
    leaky_relu = LeakyReLU(alpha=0.2)
    first_conv_block = convolution_block(input_layer=inputs,
                                         num_filters=first_layer_num_filters,
                                         kernel_size=(3, 3),
                                         activation_func=leaky_relu)
    conv_ds_block_1 = conv_downsample_block(input_layer=first_conv_block,
                                            num_filters=first_layer_num_filters * 2,
                                            kernel_size=(3, 3),
                                            activation_func=leaky_relu, max_pool_shape=(2, 2))
    conv_ds_block_2 = conv_downsample_block(input_layer=conv_ds_block_1,
                                            num_filters=first_layer_num_filters * 4,
                                            kernel_size=(3, 3),
                                            activation_func=leaky_relu, max_pool_shape=(2, 2))
    conv_ds_block_3 = conv_downsample_block(input_layer=conv_ds_block_2,
                                            num_filters=first_layer_num_filters * 8,
                                            kernel_size=(3, 3),
                                            activation_func=leaky_relu, max_pool_shape=(2, 2))

    ##### BOTTOM OF U-SHAPE #####
    bottom_conv_block = conv_downsample_block(input_layer=conv_ds_block_3,
                                              num_filters=first_layer_num_filters * 16,
                                              kernel_size=(3, 3),
                                              activation_func=leaky_relu, max_pool_shape=(2, 2))

    ###### DECODER BRANCH ######
    conv_us_block_1 = conv_upsample_block(input_layer=bottom_conv_block, skip_connection_layer=conv_ds_block_3,
                                          num_filters=first_layer_num_filters * 8,
                                          kernel_size=(3, 3),
                                          activation_func=leaky_relu, upsampling_shape=(2, 2))
    conv_us_block_2 = conv_upsample_block(input_layer=conv_us_block_1, skip_connection_layer=conv_ds_block_2,
                                          num_filters=first_layer_num_filters * 4,
                                          kernel_size=(3, 3),
                                          activation_func=leaky_relu, upsampling_shape=(2, 2))
    conv_us_block_3 = conv_upsample_block(input_layer=conv_us_block_2, skip_connection_layer=conv_ds_block_1,
                                          num_filters=first_layer_num_filters * 2,
                                          kernel_size=(3, 3),
                                          activation_func=leaky_relu, upsampling_shape=(2, 2))
    last_conv_block = conv_upsample_block(input_layer=conv_us_block_3, skip_connection_layer=first_conv_block,
                                          num_filters=first_layer_num_filters,
                                          kernel_size=(3, 3),
                                          activation_func=leaky_relu, upsampling_shape=(2, 2))

    output_layer = Conv2D(num_classes, (1, 1), activation='sigmoid')(last_conv_block)

    model = Model(inputs=[inputs], outputs=[output_layer])

    sgd_opt = SGD(lr=0.01, momentum=0.0, decay=0.0001, nesterov=False)
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-5), loss=dice_coef_loss, metrics=['accuracy'])  # decay=1e-5
    print(model.summary())

    return model


#############################################################################################################


################## TRAINING THE MODEL #######################################################################
def create_and_train_model(training_data, validation_data, input_height, input_width, first_layer_num_filters,
                           num_classes, transfer_weights=""):
    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet(input_height, input_width, first_layer_num_filters, num_classes)
    print(model.summary())
    try:
        model.load_weights(transfer_weights)
    except:
        print('Did not find existing model')

    model_checkpoint = ModelCheckpoint('cell_patches_unet_weights_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',
                                       monitor='val_loss', save_best_only=False)
    model_tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
    model_earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10)
    model_LRSchedule = LearningRateScheduler(lr_fun)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    batch_size = 8  # was 8
    num_epochs = 100
    num_aug = 2  # use imgaug lib for augmentation

    model.fit_generator(generate_training_batches(training_data, batch_size=batch_size),
                        steps_per_epoch=np.ceil(num_aug * len(training_data) / batch_size), epochs=num_epochs,
                        verbose=1,
                        validation_data=generate_training_batches(validation_data, batch_size=batch_size),
                        validation_steps=np.ceil(len(validation_data) / batch_size),
                        callbacks=[model_checkpoint, model_earlystop, model_LRSchedule, model_tensorboard])
    model.save('Daniel_Shriki_Itay_Regev_model.h5')


##############################################################################################################


################## DISPLAYING MODEL RESULTS ##################################################################
def postprocess_segmentation(segmentation_prediction):
    segmentation_prediction[segmentation_prediction >= 0.5] = 1
    segmentation_prediction[segmentation_prediction < 0.5] = 0

    return segmentation_prediction


def predict_using_patches(processed_img, trained_model, patch_rows=image_height_for_training,
                          patch_cols=image_width_for_training):
    """
    Predicts a segmentation for a single image using patch-based method.
    The input image is cropped into patches, these patches are fed into the segmentation model,
    and resulting predictions are stitched back together to make a final segmentation mask.
    :param processed_img: a processed input image (after calling preprocess_image)
    :param trained_model: a trained segmentation model
    :param patch_rows: number of rows in a patch
    :param patch_cols: number of cols in a patch
    :return: a 2-dimensional numpy array representing processed_img's cell segmentation.
    """
    img_rows, img_cols, img_channels = processed_img.shape
    segmentation_prediction = np.zeros((img_rows, img_cols))

    # loop over image and crop patches of size (patch_rows, patch_cols)
    for i in range(0, img_rows, patch_rows):
        i = max(min(i, img_rows - patch_rows - 1), 0)
        for j in range(0, img_cols, patch_cols):
            j = max(min(j, img_cols - patch_cols - 1), 0)
            img_patch = np.copy(processed_img[i:i + patch_rows, j:j + patch_cols, :])
            img_patch = np.expand_dims(img_patch, 0)
            # save segmentation result into final mask
            segmentation_prediction[i:i + patch_rows, j:j + patch_cols] = np.squeeze(trained_model.predict(img_patch))

    return segmentation_prediction


def predict_single_image(path_to_image, trained_model, display_result=True):
    """
    Predict a cell segmentation for a single image.
    :param path_to_image: the absolute path to a cell png image.
    :param trained_model: a trained model to be used for prediction.
    :param display_result: a boolean flag indicating whether results should be displayed after prediction.
    :return: a 2-dimensional numpy array of shape (height, width) representing img_array's cell segmentation.
    """
    img_array = load_image(path_to_image)
    processed_img = preprocess_image(img_array)

    segmentation_prediction = predict_using_patches(processed_img, trained_model)

    processed_segmentation = postprocess_segmentation(segmentation_prediction)

    if display_result:
        imshow(img_array)
        imshow(processed_segmentation)

    return processed_segmentation


def dice_coef_for_np_array(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    smooth = 1

    intersection_size = np.sum(y_true_flat * y_pred_flat)
    dice_score = (2. * intersection_size + smooth) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + smooth)

    return dice_score


def evaluate_model(path_to_folder, path_to_weights):
    """
    Predicts a cell segmentation for all images within a given folder (for example, the "val" folder).
    Each segmentation prediction will be save along its input image under the name "segmentationPrediction.png".
    In addition, an average dice score will be calculated over the given data set.
    :param path_to_folder: the absolute path to a folder containing cell png images, in the same structure as
            the "train" and "val" folders.
    :param path_to_weights: the absolute path to a .h5 weights file.
    :return: a floating number, representing the average dice score over the given data set.
    """
    # load trained model
    trained_model = get_unet(None, None, 32, 1)
    trained_model.load_weights(path_to_weights)

    # predict segmentation for all images in given folder
    average_dice_score = 0
    all_images_in_folder = glob.glob(path_to_folder + "/*/" + image_name)
    total_num_images = len(all_images_in_folder)

    for path_to_image in all_images_in_folder:
        print("Predicting segmentation for:", path_to_image)

        segmentation_prediction = predict_single_image(path_to_image, trained_model, display_result=False)

        save_path = os.path.join(os.path.dirname(path_to_image),
                                 os.path.basename(path_to_weights) + "_segmentationPrediction.png")
        imsave(save_path, segmentation_prediction)

        path_to_mask = path_to_image.replace(image_name, mask_name)
        mask_array = preprocess_mask(load_mask(path_to_mask))

        curr_dice_score = dice_coef_for_np_array(mask_array, segmentation_prediction)

        average_dice_score += curr_dice_score
        print("Dice score:", curr_dice_score)

    average_dice_score /= total_num_images
    return average_dice_score


##############################################################################################################


################## RUNNING OUR CODE ##########################################################################
if __name__ == '__main__':
    # Set channel ordering to match Tensorflow backend: ('channels_last' for Tensorflow, 'channels_first' for Theano)
    # K.set_image_data_format('channels_last')  # TF dimension ordering in this code
    #
    # # Get training and validation data:
    # training_data, validation_data = get_training_data(path_to_data_dir)
    # # Train model using data:
    # create_and_train_model(training_data, validation_data, input_height=None, input_width=None,
    #                        first_layer_num_filters=32, num_classes=1)

    dice_score = evaluate_model(
        path_to_folder="/media/dani94/5203b9a4-e9d7-4808-b433-8f94413806ca/project1/class2_data/val",
        path_to_weights=weight_dir)
    print("The dice score for evaulated dataset:", dice_score)

##############################################################################################################
