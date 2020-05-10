from tensorflow.keras import Model, layers
import tensorflow as tf
import Utils


def _createVocabSelection(embSize=512, vocabSize=5000):
    inLayer = layers.Input(embSize)
    vocabSelection = layers.Dense(vocabSize, 'linear')(inLayer)
    return Model(inLayer, vocabSelection)


def _createFeatureEncoder(embSize):
    encodeLayer = layers.Dense(embSize, 'relu')

    inLayer1 = layers.Input((8, 8, 2048))
    imgFeature = layers.GlobalAveragePooling2D()(inLayer1)
    f1 = encodeLayer(imgFeature)

    inLayer2 = layers.Input(2048)
    f2 = encodeLayer(inLayer2)

    return Model(inLayer1, f1), Model(inLayer2, f2)


class CaptionModel(tf.keras.Model):

    def __init__(self, tokenizer, embeddingDim, units, vocabSize, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.embeddings = tf.keras.layers.Embedding(vocabSize, embeddingDim)
        self.decoder = LSTM_Decoder(embeddingDim, units, vocabSize)
        self.imgEncoder2d, self.imgEncoder1d = _createFeatureEncoder(embeddingDim)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def greedyInferenceFromFeatures(self, features, maxLen, tokenizer=None):
        seqOutputs = []

        hidden = self.decoder.generateStartState(batch_size=features.shape[0])
        features = self.imgEncoder2d(features)
        pred, hidden = self.decoder(features, hidden)
        out = tf.argmax(pred, axis=-1)
        seqOutputs.append(out.numpy())

        for i in range(1, maxLen):
            pred, hidden = self._textInferenceStep(out, hidden)
            out = tf.argmax(pred, axis=-1)
            seqOutputs.append(out.numpy())

        if (tokenizer == None):
            return seqOutputs
        return Utils.cleanseOutputs([[tokenizer.index_word[w] for w in turn] for turn in seqOutputs])

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0
        maxLen = target.shape[1] - 1

        hidden = self.decoder.generateStartState(batch_size=target.shape[0])
        with tf.GradientTape() as tape:
            features = self.imgEncoder2d(img_tensor)
            #print("Features:", features.shape)

            # Pass image
            predictions, hidden = self.decoder(features, hidden)
            loss += self.loss_function(target[:, 0], predictions)

            # Pass text
            for i in range(maxLen):
                predictions, hidden = self._textInferenceStep(target[:, i], hidden)
                seqLoss = self.loss_function(target[:, i + 1], predictions)
                loss += seqLoss

        total_loss = loss / int(target.shape[1])

        trainable_variables = self.imgEncoder1d.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss

    def _textInferenceStep(self, textIn, hidden):
        embIn = self.embeddings(textIn)
        return self.decoder(embIn, hidden)


class LSTM_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(LSTM_Decoder, self).__init__()

        self.units = units
        self.lstm = tf.keras.layers.LSTMCell(self.units, recurrent_initializer='glorot_uniform')

        self.vocabSelection = _createVocabSelection(embedding_dim, vocab_size)
        self.vocabSelection.summary()

    def call(self, x, hidden):
        #print("X", x.shape, " Hidd:", hidden[0].shape, hidden[1].shape)
        output, states = self.lstm(x, hidden)
        return self.vocabSelection(output), states

    def generateStartState(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
