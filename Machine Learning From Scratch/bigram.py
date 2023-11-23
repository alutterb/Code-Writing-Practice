from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

'''
A bigram is an n-gram where n = 2, i.e. we are pairing up sequential words in a sentence
to capture words that frequently appear next to each other
Example - 'Entropy will kill us all.'
As a bigram - ('Entropy', 'will'), ('will', 'kill'), ('kill', 'us'), ('us', 'all')
We can then take this bigram and use a vectorizer like Tfid to see the frequency 
of the bigram amongst all documents
Note - this will be sparser than using a unigram, as the bigram likely appears less
frequently in the document. 
This also leads to sparser and higher dimensional vectors
'''


documents = ['Entropy will kill us all!', 'The only thing that can kill us is death itself.']
stop_words = stopwords.words('english')

def create_bigram(documents):
    bigram_res = []
    for doc in documents:
        res = []
        # tokenize
        words = word_tokenize(doc)
        # remove stop words and punctuation
        words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
        # create bigram
        N = len(words)
        for i in range(N):
            if i + 1 >= N:
                break
            res.append([words[i], words[i+1]])
        res = ' '.join([' '.join(word) for word in res])
        bigram_res.append(res)
    return bigram_res

def vectorize(bigrams):
    '''
    Now we can take our bigrams and convert it to a BoW model using TfidVectorizer.
    Want the input to vectorizer to be like: [['entropy', 'kill', 'kill', 'us'], ...]
    '''
    vectorizer = TfidfVectorizer()
    bow_tfid = vectorizer.fit_transform(bigrams)
    print(bow_tfid)

def main():
    bigrams = create_bigram(documents)
    print(bigrams)
    vectorize(bigrams)

if __name__ == "__main__":
    main()