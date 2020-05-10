import tensorflow as tf


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def loadImages(imagePaths):
    return tf.convert_to_tensor([load_image(p) for p in imagePaths])


def generateInceptionFeatureModel(cutLayer=-3):
    image_model = tf.keras.applications.InceptionV3(weights="imagenet")

    new_input = image_model.input
    hidden_layer = image_model.layers[cutLayer].output

    model = tf.keras.Model(new_input, hidden_layer)
    return model


def cleanseOutputs(inferenceResults):
    cleanSents = []
    for sent in zip(*[r for r in inferenceResults]):
        temp = ""
        for w in sent:
            if (w == "<end>"):
                break
            temp += w + " "
        cleanSents.append(temp.strip())
    return cleanSents
