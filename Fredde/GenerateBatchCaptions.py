import Model, TextPreProcessing, Utils, Inference
from SimpleLSTM import SimpleModel
from MS_Coco import Dataset
import tensorflow as tf
import os, pickle


def generateTrainingCaptions(model, tokenizer, batchSize=32):
    inceptionFiles = Dataset.getAllInceptionFiles()
    generatedCaptions = {}
    for f in inceptionFiles:
        print(f)
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        keys = list(data.keys())

        for i in range(0, len(keys), batchSize):
            print(i, "/", len(keys))
            fetchKeys = keys[i:i + batchSize]
            features = [data[k] for k in fetchKeys]
            features = tf.reshape(features, (len(features), 64, 2048))
            # print(features.shape)
            features = model.encoder(features)
            captions = Inference.greedyInferenceWithFeatures(model, features, tokenizer, 25)
            for j, k in enumerate(fetchKeys):
                generatedCaptions[k] = captions[j]

    with open('MS-COCO-GeneratedTrainCaptions.pkl', 'wb') as fp:
        pickle.dump(generatedCaptions, fp)


def generateEvalCaptions(model):
    featureModel = Utils.generateInceptionFeatureModel()
    mainFolder = "MS_Coco/val2017/"
    files = [(mainFolder + f, f) for f in os.listdir(mainFolder)]
    batchSize = 32
    allCaptions = {}

    for i in range(0, len(files), batchSize):
        print(i, "/", len(files))
        batchFiles = files[i:i + batchSize]
        batchCaptions = Inference.greedyInference(model, featureModel, tokenizer, [b[0] for b in batchFiles], 25)

        for c, f in zip(batchCaptions, [b[1] for b in batchFiles]):
            allCaptions[f] = c

    with open("MS-COCO-Val2017-GeneratedCaptions-SimpleModel0.4.pkl", 'wb') as fp:
        pickle.dump(allCaptions, fp)


if __name__ == '__main__':
    img2seqs, tokenizer = TextPreProcessing.generateCaptionData()
    '''
    model = Model.CaptionModel(tokenizer, 256, 512, 5000)
    model.load_weights("ImageCaptionModel-Weights")
    '''
    model = SimpleModel.CaptionModel(tokenizer, 512, 512, 5000)
    model.load_weights('LSTM-ImageCaptionModel-Weights')

    generateEvalCaptions(model)
    # generateTrainingCaptions(model, tokenizer)
