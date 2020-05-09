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

from ImageCaptioning import Models

VOCAB_SIZE = 5000
DATASET_PATH = "/../../Dataset"


def downloadData(numImages=-1):
    # Download caption annotation files
    annotation_folder = DATASET_PATH + '/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    # Download image files
    image_folder = DATASET_PATH + '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    annotation_file = os.path.abspath('.') + DATASET_PATH + '/annotations/captions_train2014.json'

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


# "A man throwing a frisbee." => ["A", "man", "throwing", "a", "frisbee"]
def preProcessTokenize(train_captions):
    # print(train_captions[0])
    # Choose the top 5000 words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # print(train_seqs[0])

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    # max_length = calc_max_length(train_seqs)
    return cap_vector


# Load the numpy files, used in map
def loadNumpyFiles(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def preProcessData():
    numImages = -1
    trainCaptions, imageNameVector = downloadData(numImages)
    # print(len(train_captions))
    # print(len(img_name_vector))
    inceptionNet = Models.getExtractModel()
    # print(img_name_vector[0])
    # extractModel(img_name_vector[0])
    # cacheFeatures(inceptionNet, img_name_vector)
    preProcessedCaptions = preProcessTokenize(trainCaptions)
    # Create training and validation sets using an 80-20 split
    imgNameTrain, imgNameVal, capTrain, capVal = train_test_split(imageNameVector, preProcessedCaptions,
                                                                  test_size=0.2, random_state=0)
    print(len(imgNameTrain), len(capTrain), len(imgNameVal), len(capVal))

    # Feel free to change these parameters according to your system's configuration

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = VOCAB_SIZE + 1
    num_steps = len(imgNameTrain) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((imgNameTrain, capTrain))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        loadNumpyFiles, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return inceptionNet, dataset
