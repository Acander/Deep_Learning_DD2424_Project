from SimpleLSTM import SimpleModel
from MS_Coco import Dataset
import TextPreProcessing
import tensorflow as tf
import numpy as np
import pickle
import Model


def _selectRandomCaption(captions):
    return captions[np.random.choice(range(len(captions)))]


def generateFetchData(fetchKeys, data, img2Captions):
    imgs, texts = [], []
    maxChar = 0
    for k in fetchKeys:
        if (k in img2Captions):
            imgs.append(data[k])
            txt = _selectRandomCaption(img2Captions[k])
            texts.append(txt)

            # endChar = np.where(txt == 0)[0][0]
            # maxChar = max(maxChar, endChar)

    f = lambda x: tf.convert_to_tensor(x)
    return f(imgs), f(texts)


def trainOnInceptionFile(filePath, img2Captions, model, batchSize=16, epoch='0'):
    with open(filePath, 'rb') as fp:
        data = pickle.load(fp)
    keys = list(data.keys())
    np.random.shuffle(keys)

    windowSize = 100
    lossWindow = []
    for i in range(0, len(keys), batchSize):
        fetchKeys = keys[i:i + batchSize]
        inImgs, inTexts = generateFetchData(fetchKeys, data, img2Captions)
        if (len(inImgs) > 0):
            # print(inTexts[0])
            loss, std = model.train_step(inImgs, inTexts)

            lossWindow.insert(0, loss.numpy())
            lossWindow = lossWindow[:windowSize]
            print("Epoch: {} - Loss: {}  -  STD: {}".format(epoch, np.mean(lossWindow), std))


if __name__ == '__main__':
    inceptionFiles = Dataset.getAllInceptionFiles()
    img2seqs, tokenizer = TextPreProcessing.generateCaptionData()
    # model = Model.CaptionModel(tokenizer, 256, 512, 5000)
    model = SimpleModel.CaptionModel(tokenizer, 512, 512, 5000)
    # model.load_weights('LSTM-ImageCaptionModel-Weights')

    epoch = 0
    while (True):
        for i, f in enumerate(inceptionFiles):
            trainOnInceptionFile(f, img2seqs, model, epoch="{}/{} E:{}".format(i + 1, len(inceptionFiles), epoch))

        model.save_weights("LSTM-ImageCaptionModel-NoMaxLen-Weights")
        epoch += 1
