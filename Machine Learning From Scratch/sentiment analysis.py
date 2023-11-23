from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

'''
1. Preprocess text to remove stopwords, lemmatize, etc..
2. Vectorize text to create bag of worlds model to represent data
3. Build and train sentiment analysis model
'''

# 0 - negative | 1 - positive
documents_with_labels = [('I am so angry right now! I could destroy the entire world with my fists!', 0), 
                         ("I am so happy right now!", 1),
                         ("How dare you try to steal my wallet!", 0), 
                         ("That's right! You are my best friend.", 1)]

class SentimentAnalysis():
    def __init__(self,data):
        '''
        Input: List of tuples. Each tuple represents a document.
        First entry in the tuple is the sentence.
        Second entry is positive (1) or negative (0) sentiment associated with the sentence
        '''
        self.data = data
        self.vectorizer = None
        self.documents = [tup[0] for tup in self.data]
        self.labels = [tup[1] for tup in self.data] # labels - what we are trying to predict
        self.NB = None # We will use Naive Bayes as the model for prediction

    @staticmethod
    def preprocess(documents):
        '''
        Tokenizes text, removes stopwords, and lemmatizes.
        Inherently, CountVectorizer can handle stopwords and punctuation,
        but for proof of concept we will use the nltk libraries
        '''
        lemmatizer = WordNetLemmatizer()
        docs_tokenized = [word_tokenize(doc) for doc in documents]
        stop_words = stopwords.words('english')
        docs_filtered = []
        for doc in docs_tokenized:
            words_filtered = [lemmatizer.lemmatize(w)
                               for w in doc if w.lower() not in stop_words and w.isalnum()]
            docs_filtered.append(words_filtered)
        return docs_filtered

    def vectorize(self, documents):
        '''
        Pass in preprocessed data to a bag of words model.
        Here we will use TfidVectorizer
        '''

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(documents)
        return X

    def train(self):
        docs_filtered = self.preprocess(self.documents)
        docs_filtered = [' '.join(doc) for doc in docs_filtered]
        X = self.vectorize(docs_filtered)
        X_train, X_test, y_train, y_test = train_test_split(X, self.labels, test_size=0.2)
        self.NB = MultinomialNB()
        self.NB.fit(X_train, y_train)

        y_pred = self.NB.predict(X_test)
        print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")


    def predict(self, text):
        '''
        Input: New document string without labels
        Output: Predicted class
        '''
        # convert to list for data consistency
        text = [text]
        # preprocess
        docs_filtered = self.preprocess(text)
        docs_filtered = [' '.join(doc) for doc in docs_filtered]
        # vectorize using same vectorizing model
        X = self.vectorizer.transform(docs_filtered)
        # predict with trained model
        pred_class = self.NB.predict(X)
        print(f"Predicted class: {pred_class}")

def main():
    SA = SentimentAnalysis(documents_with_labels)
    SA.train()
    SA.predict('I am so angry right now. I could destroy the world!')


if __name__ == "__main__":
    main()