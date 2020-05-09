import tensorflow as tf


def getExtractModel():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    # new_input = image_model.input
    # hidden_layer = image_model.layers[-1].output

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # image_features_extract_model = tf.keras.Model(new_input, hidden_layer)  # , global_average_layer)
    image_features_extract_model = tf.keras.Sequential([image_model, global_average_layer])
    return image_features_extract_model
