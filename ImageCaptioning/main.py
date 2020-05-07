import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import json
from glob import glob
from PIL import Image
import pickle


def downloadData(numImages=-1):
    # Download caption annotation files
    annotation_folder = '/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    # Download image files
    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    annotation_file = 'annotations/captions_train2014.json'

    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

    # Select the first 30000 captions from the shuffled set
    train_captions = train_captions[:numImages]
    img_name_vector = img_name_vector[:numImages]
    return train_captions, img_name_vector


def loadImage(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def getExtractModel():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # new_input = image_model.input
    # hidden_layer = image_model.layers[-1].output

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # image_features_extract_model = tf.keras.Model(new_input, hidden_layer)  # , global_average_layer)
    image_features_extract_model = tf.keras.Sequential([image_model, global_average_layer])
    return image_features_extract_model


def cacheFeatures(extractModel, imageNameVec):
    # Get unique images
    encode_train = sorted(set(imageNameVec))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(loadImage, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    batchCounter = 0
    for img, path in image_dataset:
        batchCounter += 1
        if batchCounter % 100 == 0:
            print("Cached images:", batchCounter * 16)
        # print(img.shape)
        batch_features = extractModel(img)
        # print(batch_features.shape)
        # batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def run():
    numImages = -1
    train_captions, img_name_vector = downloadData(numImages)
    # print(len(train_captions))
    # print(len(img_name_vector))
    extractModel = getExtractModel()
    # print(img_name_vector[0])
    # extractModel(img_name_vector[0])
    # cacheFeatures(extractModel, img_name_vector)


if __name__ == '__main__':
    run()
