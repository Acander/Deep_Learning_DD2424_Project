from MS_Coco import Dataset
import tensorflow as tf
import pickle


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def generateCaptionData():
    print("Loading Captions")
    allCaptions, caption2Img = Dataset.getAllAnnotations()

    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(allCaptions)
    train_seqs = tokenizer.texts_to_sequences(allCaptions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    img2seqs = {}
    for capt, seq in zip(allCaptions, cap_vector):
        img = caption2Img[capt]
        if (img not in img2seqs):
            img2seqs[img] = []
        img2seqs[img].append(seq)

    return img2seqs, tokenizer


def generateBertTokenizedCaptions():
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    print("Loading Captions")
    allCaptions, caption2Img = Dataset.getAllAnnotations("captions_val2017.json")
    train_seqs = [tokenizer.encode(capt) for capt in allCaptions]

    img2seqs = {}
    for capt, seq in zip(allCaptions, train_seqs):
        img = caption2Img[capt]
        if (img not in img2seqs):
            img2seqs[img] = []
        img2seqs[img].append(seq)

    return img2seqs


if __name__ == '__main__':
    '''
    img2, tokenizer = generateCaptionData()
    with open("Ms-Coco-Annotations-5kVocab.pkl", 'wb') as fp:
        pickle.dump((img2, tokenizer), fp)
    '''

    img2Seq = generateBertTokenizedCaptions()
    with open("MS-Coco-eval2017-Img2BertInds.pkl", 'wb') as fp:
        pickle.dump(img2Seq, fp)
