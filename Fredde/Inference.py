from SimpleLSTM import SimpleModel
import TextPreProcessing
import tensorflow as tf
import Utils, Model
import numpy as np
import Utils


def generateImgFeatures(featureModel, imgPaths):
    temp_input = tf.convert_to_tensor([Utils.load_image(p) for p in imgPaths])
    return featureModel(temp_input)


def _beamSearchImage(model, features, tokenizer, maxLen, beamSize=3):
    dec_input = tf.convert_to_tensor([tokenizer.word_index['<start>'] for _ in range(beamSize)])
    hidden = model.decoder.reset_state(batch_size=beamSize)
    features = tf.broadcast_to(features, (beamSize,) + features.shape)

    results, propScores = [], [0] * beamSize
    for i in range(maxLen):
        dec_input = tf.expand_dims(dec_input, axis=1)
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)
        if (i == 0):
            predicted_id = np.argsort(predictions[0])[-beamSize:]
        else:
            predicted_id = np.argmax(predictions, axis=-1)

        for i, pred in enumerate(predicted_id):
            propScores[i] += predictions[i][pred].numpy()

        results.append([tokenizer.index_word[w] for w in predicted_id])
        dec_input = tf.convert_to_tensor(predicted_id)

    results = [(t, p) for t, p in zip(Utils.cleanseOutputs(results), propScores)]
    return sorted(results, key=lambda x: x[1], reverse=True)


def beamSearchImages(model, featureModel, tokenizer, imgPaths, maxLen):
    features = generateImgFeatures(model, featureModel, imgPaths)
    results = [_beamSearchImage(model, f, tokenizer, maxLen) for f in features]
    return results


def greedyInferenceWithFeatures(model, features, tokenizer, maxLen):
    dec_input = tf.convert_to_tensor([tokenizer.word_index['<start>'] for _ in features])
    hidden = model.decoder.reset_state(batch_size=len(features))

    results = []
    for i in range(maxLen):
        dec_input = tf.expand_dims(dec_input, axis=1)
        predictions, hidden, attention_weights = model.decoder(dec_input, features, hidden)

        predicted_id = np.argmax(predictions, axis=-1)
        results.append([tokenizer.index_word[w] for w in predicted_id])
        dec_input = tf.convert_to_tensor(predicted_id)

    return Utils.cleanseOutputs(results)


def greedyInference(model, featureModel, tokenizer, imgPaths, maxLen):
    features = generateImgFeatures(featureModel, imgPaths)
    return model.greedyInferenceFromFeatures(features, maxLen, tokenizer)
    # return greedyInferenceWithFeatures(model, features, tokenizer, maxLen)


if __name__ == '__main__':
    img2seqs, tokenizer = TextPreProcessing.generateCaptionData()
    '''
    model = Model.CaptionModel(tokenizer, 256, 512, 5000)
    model.load_weights("ImageCaptionModel-Weights")
    '''
    model = SimpleModel.CaptionModel(tokenizer, 512, 512, 5000)
    model.load_weights('LSTM-ImageCaptionModel-Weights')
    featureModel = Utils.generateInceptionFeatureModel()

    imgPaths = ["Fredrik 2020.png", "Petra.jpg", "Mange.jpg", "Träd Dude.png", "cars.jpg", "lions.jpg"]
    # imgPaths = ['Joey.png', 'Joey2.png', 'Adrian1.png', 'Adrian2.png', 'Joey3.png']
    features = generateImgFeatures(featureModel, imgPaths)
    results = model.greedyInferenceFromFeatures(features, 20, tokenizer)
    for r in results:
        print(r)

    '''
    # for i in range(10):
    res = greedyInference(model, featureModel, tokenizer, imgPaths, 20)
    print(res)
    res2 = beamSearchImages(model, featureModel, tokenizer, imgPaths, 20)
    print(res2)
    '''
