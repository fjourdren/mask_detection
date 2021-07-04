import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

import os
import time
import matplotlib.pyplot as plt
import numpy as np


# train params
epochs     = 20
batch_size = 10

# image size
img_size = (64, 64) # target image size


# saving path
model_root_path = os.path.join("model")

# dataset paths
path_train      = os.path.join("dataset", "train")
path_validation = os.path.join("dataset", "validation")
path_test       = os.path.join("dataset", "test")

# init dataset vars to have access everywhere
train_generator      = None
validation_generator = None

def prepare_dataset():
    print("=== Preparing dataset ===")
    datagen = ImageDataGenerator(rescale=1./255) # rescale RGB on base 255 in a float base (value/255)

    # put dataset in global var to use it in other functions
    global train_generator
    global validation_generator

    # training generator
    train_generator = datagen.flow_from_directory(
        directory=path_train, 
        target_size=img_size, 
        class_mode='binary',
        batch_size=batch_size, 
        shuffle=True
    )

    # test generator
    validation_generator = datagen.flow_from_directory(
        directory=path_test, 
        target_size=img_size, 
        class_mode='binary',
        batch_size=batch_size, 
        shuffle=True
    )

def build_model(nbClasses):
    print("=== Model builder ===")
    print("Building model...")

    # model params
    kernel_conv_size = (3, 3) # convolution kernel size
    kernel_pool_size = (2, 2) # pool kernel size
    dropout_value = 0.25 # dropout layer value
    input_shape = (img_size[0], img_size[1], 3) # 64x64 images with 3 channels
    
    # model convolution
    model = keras.Sequential()

    model.add(Conv2D(32, kernel_conv_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_conv_size, activation='relu'))
    model.add(MaxPooling2D(kernel_pool_size))
    model.add(Dropout(dropout_value))

    model.add(Conv2D(32, kernel_conv_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_conv_size, activation='relu'))
    model.add(MaxPooling2D(kernel_pool_size))
    model.add(Dropout(dropout_value))

    # detection network
    model.add(Flatten())  # converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))

    # last layer (add number of layers equals to number of categories) => output in one hot encoding
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
        optimizer='adam'
    )

    print("Model built !")
    
    return model
    
def train_model(model):
    print("=== Training ===")
    # prepare training 
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) # early stopping config (3 validation test result under the previous perfs

    # training
    print("Training...")
    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.n//train_generator.batch_size, # traditionally, the steps per epoch is calculated as train_length // batch_size => adapt steps per epoch to data size and make batch_size become the number of steps to do between each back propagation steps
        validation_data=validation_generator,
        validation_steps=validation_generator.n//validation_generator.batch_size, # traditionally, the steps per epoch is calculated as train_length // batch_size => adapt steps per epoch to data size and make batch_size become the number of steps to do between each back propagation steps
        verbose=2,
        callbacks=[early_stopping] # early stopping config
    )

    train_time = round((time.time() - start_time) / 60, 3) # get train time in minute
    print("Trained in %smin." % (train_time))

    return history

def save_model(model, path):
    print("=== Model saving ===")
    model.save(path)
    print("Model saved in {}".format(path))

def save_training_history(history):
    print("=== Saving history ===")

    # calculate intervals
    interval = np.arange(0, len(history.history["loss"]))  # an interval per epoch


    # === make loss plot ===
    plt.figure()

    # add data
    plt.plot(interval, history.history["loss"], label="train_loss")
    plt.plot(interval, history.history["val_loss"], label="val_loss")

    # labels
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    # save plot on disk
    plt.savefig(os.path.join(model_root_path, "loss_history.png"))
    print("loss_history.png saved")



    # === make accuracy plot ===
    plt.figure()

    # add data
    plt.plot(interval, history.history["binary_accuracy"], label="train_acc")
    plt.plot(interval, history.history["val_binary_accuracy"], label="val_acc")

    # labels
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # save plot on disk
    plt.savefig(os.path.join(model_root_path, "accuracy_history.png"))
    print("accuracy_history.png saved")


def run():
    prepare_dataset() # make generators

    nbClasses = len(train_generator.class_indices) # get number of classes in the train data
    model = build_model(nbClasses) # build model with nbClasses output neurons

    history = train_model(model) # train model

    os.makedirs(model_root_path, exist_ok=True) # be sure that output dir exist

    save_model(model, os.path.join(model_root_path, 'model.h5')) # save model on disk
    save_training_history(history) # show history to the user

if __name__ == "__main__":
    run()