import cv2
import glob
import time
import numpy as np
import random
import torch
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

from eventvision import read_dataset, read_annotation


def makeModel(inputs, outputsize):
    """ Makes a simple object detection model compatible with tensorflow. """
    x = layers.Conv2D(filters=16, kernel_size=(5,5),padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)
    
    x = layers.Conv2D(filters=32, kernel_size=(5,5),padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)
    
    x = layers.Conv2D(filters=64, kernel_size=(5,5),padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)
    
    x = layers.Conv2D(filters=128, kernel_size=(5,5),padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)
    
    x = layers.Conv2D(filters=256, kernel_size=(5,5),padding="same")(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(units=2048)(x)
    x = layers.Dense(units=1024)(x)
    outputs = layers.Dense(units=outputsize)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="imageToBB")
    #model.summary()
    
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    return model


def parse_total_dataset(getData, getAnnotation, event=True):
    """ Parses the entire Caltech101 dataset.
        When event=True the events and annotations of the NCaltech101 dataset are parsed.
        When event=False the images and annotations of the Caltech101 dataset are parsed.
        Returns X_train, X_test, y_train, y_test
    """
    train_points = []
    train_labels = []
    if event:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_annotations\*')
    else:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_objects\101_ObjectCategories\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_annotations\Annotations\*')
        
    length = len(event_folders)

    for i, _ in enumerate(event_folders):
        events = glob.glob(event_folders[i] +"\\*")
        annotation = glob.glob(ann_folders[i] + "\\*")
        
        for j, _ in enumerate(events):
            w, h, mtrx = getData(events[j])
            minX, minY, maxX, maxY = getAnnotation(annotation[j])
            
            label = np.zeros(length + 4)
            label[-4:] = [minX / w, minY / h, maxX / w, maxY / h]
            label[i] = 1.0
            
            train_points.append(mtrx)
            train_labels.append(label)
            
    train_points = np.array(train_points, dtype="float32")
    train_labels = np.array(train_labels, dtype="float32")
    
    return train_test_split(train_points, train_labels, test_size=0.2, random_state=42)


def parse_partial_dataset(getData, getAnnotation, event=True, folders=["airplanes", "car_side", "helicopter", "Motorbikes"]):
    """ Parses a part of the Caltech101 dataset.
        When event=True the events and annotations of the NCaltech101 dataset are parsed.
        When event=False the images and annotations of the Caltech101 dataset are parsed.
        The folders argument denotes which classes are parsed.
        Returns X_train, X_test, y_train, y_test
    """
    train_points = []
    train_labels = []
    if event:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_annotations\*')
    else:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_objects\101_ObjectCategories\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_annotations\Annotations\*')
    
    event_folders = [s for s in event_folders if any(xs in s for xs in folders)]
    ann_folders = [s for s in ann_folders if any(xs in s for xs in folders)]
    
    length = len(event_folders)

    for i, _ in enumerate(event_folders):
        events = glob.glob(event_folders[i] +"\\*")
        annotation = glob.glob(ann_folders[i] + "\\*")
        
        for j, _ in enumerate(events):
            w, h, mtrx = getData(events[j])
            minX, minY, maxX, maxY = getAnnotation(annotation[j])
            
            label = np.zeros(length + 4)
            label[-4:] = [minX / w, minY / h, maxX / w, maxY / h]
            label[i] = 1.0
            
            train_points.append(mtrx)
            train_labels.append(label)
            
    train_points = np.array(train_points, dtype="float32")
    train_labels = np.array(train_labels, dtype="float32")
    
    return train_test_split(train_points, train_labels, test_size=0.2, random_state=42)


def parse_random_dataset(getData, getAnnotation, event=True, size=1000, folders=["airplanes", "car_side", "helicopter", "Motorbikes"]):
    """ Parses the entire Caltech101 dataset.
        When event=True the events and annotations of the NCaltech101 dataset are parsed.
        When event=False the images and annotations of the Caltech101 dataset are parsed.
        The size argument denotes how many samples are taken.
        The folders argument denotes which classes are parsed.
        Returns X_train, X_test, y_train, y_test
    """
    train_points = []
    train_labels = []
    if event:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_annotations\*')
    else:
        event_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_objects\101_ObjectCategories\*')
        ann_folders = glob.glob(os.getcwd() + r'\..\Caltech101_real\caltech101_annotations\Annotations\*')
    
    event_folders = [s for s in event_folders if any(xs in s for xs in folders)]
    ann_folders = [s for s in ann_folders if any(xs in s for xs in folders)]
    
    length = len(event_folders)

    for _ in range(size):
        i = random.randint(0, length - 1)
        events = glob.glob(event_folders[i] +"\\*")
        annotation = glob.glob(ann_folders[i] + "\\*")
        
        j = random.randint(0, len(events) - 1)
        
        w, h, mtrx = getData(events[j])
        minX, minY, maxX, maxY = getAnnotation(annotation[j])
        
        label = np.zeros(length + 4)
        label[-4:] = [minX / w, minY / h, maxX / w, maxY / h]
        label[i] = 1.0
        
        train_points.append(mtrx)
        train_labels.append(label)
            
    train_points = np.array(train_points, dtype="float32")
    train_labels = np.array(train_labels, dtype="float32")
    
    return train_points, train_labels


def tf_iou(box_a, box_b):
    """ Returns the iou values from the tensors box_a and box_b for tensorflow. """
    x_left = tf.math.maximum(box_a[:, 0], box_b[:, 0])
    y_top = tf.math.maximum(box_a[:, 1], box_b[:, 1])
    x_right = tf.math.minimum(box_a[:, 2], box_b[:, 2])
    y_bottom = tf.math.minimum(box_a[:, 3], box_b[:, 3])
    
    inter = (x_right - x_left) * (y_bottom - y_top)

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])) 
    union = area_a + area_b - inter
                       
    return inter / union

def torch_iou(box_a, box_b):
    """ Returns the iou values from the tensors box_a and box_b for pytorch. """
    
    x_left = torch.max(box_a[:, 0], box_b[:, 0])
    y_top = torch.max(box_a[:, 1], box_b[:, 1])
    x_right = torch.min(box_a[:, 2], box_b[:, 2])
    y_bottom = torch.min(box_a[:, 3], box_b[:, 3])
    
    inter = (x_right - x_left) * (y_bottom - y_top)

    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])) 
    union = area_a + area_b - inter
                       
    return inter / union
