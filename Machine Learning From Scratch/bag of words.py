import numpy as np
from collections import defaultdict
import string
import re

documents = ['The quick brown fox jumped over the lazy dog', 
             'this is another short document', 
             'and, guess what! This is another short document!']

def clean_word(word):
    pattern = r'[{}]'.format(re.escape(string.punctuation)) # look at all punctuation
    return re.sub(pattern, '', word).lower() # replace with empty string and lowercase it

vocab = set().union(*[set(clean_word(word) for word in doc.split()) for doc in documents])

def term_freq(doc):
    tf_dict = {k : 0 for k in vocab}
    words = doc.split()
    words = [clean_word(word) for word in words]
    N = len(words)
    for word in words:
        tf_dict[word] += 1
    tf_dict = {k : v / N for (k,v) in zip(tf_dict.keys(), tf_dict.values())}
    return tf_dict

def inv_data_freq(docs):
    N = len(docs)
    id_dict  = {k : 0 for k in vocab}
    for word in vocab:
        for doc in docs:
            if word in doc:
                id_dict[word] += 1
    id_dict = {k : np.log(N / v) if v != 0 or N != 0 else 0 for (k,v) in zip(id_dict.keys(),id_dict.values())}
    return id_dict
    
def tfidf(docs):
    tfidf_dict = defaultdict(dict)
    idf = inv_data_freq(docs)
    for idx, doc in enumerate(docs):
        tf = term_freq(doc)
        for word in vocab:
            tfidf_dict[idx][word] = idf[word] * tf[word]
    return tfidf_dict

def create_bag_of_words():
    arr = np.zeros(shape=(len(vocab), len(documents)))
    tfidf_df = tfidf(documents)
    for idx,_ in  enumerate(documents):
        for jdx,word in enumerate(vocab):
            arr[jdx,idx] = tfidf_df[idx][word]
    return arr

def main():
    bow_model = create_bag_of_words()
    print(bow_model)
    print(bow_model.shape)

if __name__ == "__main__":
    main()