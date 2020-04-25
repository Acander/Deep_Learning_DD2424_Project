# Tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

IMG_SIZE = 160  # All images will be resized to 160x160

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def run():
    # LOAD AND PRE-PROCESS DATA
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        # split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        split=[
            tfds.Split.TRAIN.subsplit(tfds.percent[:80]),
            tfds.Split.TRAIN.subsplit(tfds.percent[80:90]),
            tfds.Split.TRAIN.subsplit(tfds.percent[90:])
        ],
        with_info=True,
        as_supervised=True,
    )

    print(raw_train)
    print(raw_validation)
    print(raw_test)

    get_label_name = metadata.features['label'].int2str

    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
        plt.show()

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    for image_batch, label_batch in train_batches.take(1):
        pass

    print(image_batch.shape)

    # CREATE BASE MODEL
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False
    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)


if __name__ == '__main__':
    run()
