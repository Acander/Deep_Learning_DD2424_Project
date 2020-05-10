import tensorflow as tf
import pickle
import Utils

if __name__ == '__main__':
    from MS_Coco import Dataset

    model = Utils.generateInceptionFeatureModel()
    imgPaths = Dataset.getAllImagePaths()
    dataSize = len(imgPaths)

    batchSize = 16
    batchImgs = {}
    for i in range(0, len(imgPaths), batchSize):
        print(i, "/", dataSize)

        data = imgPaths[i:i + batchSize]
        paths, names = [d[0] for d in data], [d[1] for d in data]
        imgs = Utils.loadImages(paths)
        features = model(imgs)
        for j, f in enumerate(features):
            batchImgs[names[j]] = f

        if (len(batchImgs) >= 10000):
            with open("MS-COCO-Train-3DInception-{}.pkl".format(i), 'wb') as fp:
                pickle.dump(batchImgs, fp)
            batchImgs = {}

    with open("MS-COCO-Train-3DInception-{}.pkl".format(i), 'wb') as fp:
        pickle.dump(batchImgs, fp)

    '''
    for i, data in enumerate(image_dataset):
        print(i, "/", dataSize)
        img, path = data

        batch_features = model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
    '''
