'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
'''
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
Steps to prepare text data for a sequential model:
1. Tokenize text
2. Map these tokens to unique integers
3. Create sequences of integers from the tokens
4. Pad the sequences to be all the same length
5. Feed it the neural network
'''
documents = ['This is a document, nice', 'this is another document, great', 
             'this sentence is not like The last two.']
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    '''
    Input: list of documents, where each document is a string of words
    Output: List of preprocessed tokens - e.g. - [['document', 'nice'],
                                                  ['another', 'document', 'great],
                                                   ...]
    '''
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word.lower()) for word in text 
                                                if word.lower() not in stop_words
                                                                and word.isalnum()]
    return text

def build_vocabulary(tokenized_documents):
    '''
    Input: List of preprocessed tokens from tokenize
    Output: dictionary of vocabulary uniquely mapped to integers
    - e.g. - {'this' : 1, 'document' : 2, 'nice' : 3, 'another' : 4, 'great' : 5, ...}
    '''
    vocab = []
    vocab_mapping = {}
    for doc in tokenized_documents:
        for token in doc:
            if token not in vocab:
                vocab.append(token)
            else:
                continue
    count = 1
    for word in vocab:
        vocab_mapping[word] = count
        count += 1
    
    return vocab_mapping

def build_sequences(tokenized_documents, vocab_mapping):
    sequences = []
    for document in tokenized_documents:
        res = []
        for token in document:
            res.append(vocab_mapping[token])
        sequences.append(res)
    return sequences

def pad_sequences(sequences):
    max_length = 0
    for sequence in sequences:
        if len(sequence) > max_length:
            max_length = len(sequence)

    padded_sequences = []
    for sequence in sequences:
        while len(sequence) < max_length:
            sequence.append(0)
        padded_sequences.append(sequence)
    return padded_sequences

def main():
    tokenized_documets = [tokenize(doc) for doc in documents]
    vocab_mapping = build_vocabulary(tokenized_documets)
    sequences = build_sequences(tokenized_documets, vocab_mapping)
    padded_sequences = pad_sequences(sequences)

if __name__ == "__main__":
    main()