from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from nltk.tokenize import word_tokenize

sentences = [['Hello. I am the first sentence of this example. I hope I am not preprocessed.'],
             ['Indeed, I am the SECOND sentence. I do entirely hope to be preprocessed'],
             ['Those fools. I wish to neither be preprocessed nor left alone!']]


def train_word2vec(sentences):
    sentences = [word_tokenize(sentence[0].lower()) for sentence in sentences]
    res_sentences = []
    for sentence in sentences:
        res = []
        for word in sentence:
            if word.isalnum():
                res.append(word)
        res_sentences.append(res)

    model = Word2Vec(sentences=res_sentences, min_count=1)
    words = list(model.wv.index_to_key)
    for word in words:
        print(f"Word: {word}")
        print(f"Vector representation: {model.wv[word]}")

def main():
    train_word2vec(sentences)

if __name__ == "__main__":
    main()