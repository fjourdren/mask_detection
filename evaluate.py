import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

import os
import time

from plot_utils import render_confusion_matrix

# image size
img_size = (64, 64) # target image size


# model path
model_name = 'model.h5'
model_root_path = os.path.join("model")

# dataset paths
path_test = os.path.join("dataset", "test")

# test generator
test_generator = None


def prepare_dataset():
    print("=== Preparing dataset ===")
    datagen = ImageDataGenerator(rescale=1./255) # rescale RGB on base 255 in a float base (value/255)

    # put dataset in global var to use it in other functions
    global test_generator

    # test generator
    test_generator = datagen.flow_from_directory(
        directory=path_test, 
        target_size=img_size, 
        class_mode='binary',
        shuffle=False
    )


def load_model(model_name):
    print("=== Loading model ===")
    return keras.models.load_model(os.path.join(model_root_path, model_name))


def evaluate(model, labels):
    print("=== Evaluation ===")
    print("Test dataset in progress...")
    start_time = time.time()

    # computation
    Y_pred_proba = model.predict(test_generator) # make predictions on test dataset
    y_pred_index = np.concatenate(Y_pred_proba, axis=0)
    y_pred_index = np.round(y_pred_index, 0)
    y_pred_index = np.int_(y_pred_index)

    # execution time calculation
    execution_time = round((time.time() - start_time) / 60, 3) # get execution time in minute
    seconds_per_image = (execution_time/len(test_generator.filenames))
    imgs_per_second = 1/seconds_per_image
    print("Evaluated %d images in %smin (%fs/img or %fimgs/s)." % (len(test_generator.filenames), execution_time, seconds_per_image, imgs_per_second))

    # transform predicted class index to class name
    y_pred = []
    for pred_index in y_pred_index:
        y_pred.append(labels[pred_index])

    # transform waited class index to class name
    y_true = []
    for real_cls_index in test_generator.classes:
        y_true.append(labels[real_cls_index])

    print(y_pred)
    print(y_true)

    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate recall
    recall = recall_score(y_true, y_pred, average=None)

    # calculate precision
    precision = precision_score(y_true, y_pred, average=None)

    # calculate F1 score
    f1 = f1_score(y_true, y_pred, average=None)

    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # make report
    report = classification_report(y_true, y_pred, target_names=labels)

    # return model.evaluate(test_generator) => old evaluation system, commented because give less informations
    return accuracy, recall, precision, f1, conf_matrix, report


def show_results(model, accuracy, recall, precision, f1, conf_matrix, report, labels):
    # show model metrics
    print("- Classes metrics:")
    print(report)

    print("- Model metrics:")
    print("    * Model accuracy:  %f" % accuracy)
    print("    * Model recall:    %f" % np.mean(recall))
    print("    * Model precision: %f" % np.mean(precision))
    print("    * Model f1:        %f" % np.mean(f1))

    # render confusion matrix to the user
    render_confusion_matrix(conf_matrix, labels)


def predict_image(model, filename, labels):
    # load image
    image = load_img(path=filename, grayscale=False, color_mode='rgb', target_size=(64, 64, 3))

    # pass to array and reshape array
    image_array = img_to_array(image)
    image_array = image_array.reshape((-1, 64, 64, 3))

    # model predictions
    start_time = time.time()
    y_prob = model.predict(image_array)
    execution_time = round((time.time() - start_time) / 60, 3) # get execution time in minute

    # keep only the biggest softmax value which is the final prediction for that image
    y_classes = y_prob.argmax(axis=-1)
    
    return labels[y_classes[0]], execution_time # return class name


def predict_raw_image(model, image, labels):
    # pass to array and reshape array
    image_array = img_to_array(image)
    image_array = image_array.reshape((-1, 64, 64, 3))

    # model predictions
    start_time = time.time()
    y_prob = model.predict(image_array)
    execution_time = round((time.time() - start_time) / 60, 3) # get execution time in minute

    # keep only the biggest softmax value which is the final prediction for that image
    y_classes = y_prob.argmax(axis=-1)
    
    return labels[y_classes[0]], execution_time # return class name


def run():
    # init & load model
    prepare_dataset() # make generators
    labels = list(test_generator.class_indices.keys()) # class labels
    model = load_model(model_name)


    # predict a random image
    filename = os.path.join(path_test, np.random.choice(test_generator.filenames)) # take a random image to process in the test dataset (high probability of bad classification)
    expected_cls = filename.split(os.sep)[-2] # get random image class
    print("Performing image: %s" % filename)
    print("Expected class: %s" % expected_cls)
    predicted_class, execution_time = predict_image(model, filename, labels)
    print("Predicted class: %s" % predicted_class)
    print("Predicted in %fmin." % execution_time)


    # evaluation
    accuracy, recall, precision, f1, conf_matrix, report = evaluate(model, labels) # evaluate model


    # show evaluation results / metrics
    show_results(model, accuracy, recall, precision, f1, conf_matrix, report, labels) # show results to user


if __name__ == "__main__":
    run()