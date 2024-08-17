from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


class NaiveBayes():
    def __init__(self):
        stop_words = ['the', 'and', 'is', 'a', 'or', 'this', 'be', 'for']
        self.vectorizer = CountVectorizer(stop_words=stop_words)
        self.data = None
        self.priors = None
        self.cond_probs = None

    def fit(self, data):
        '''
        1. Generate BoW model from data
        2. Find prior probability of each class
        3. Find conditional probability of each word P(word | class)
        '''
        documents = [doc[0] for doc in data]
        self.data = self.vectorizer.fit_transform(documents) # 1.

        priors = defaultdict(float)
        for example in data:        
            label = example[1]
            priors[label] += 1
        total_documents = len(documents)
        priors = {k : v / total_documents for (k,v) in priors.items()} # 2.

        cond_probs = defaultdict(dict) # 3.
        vocabulary = self.vectorizer.vocabulary_
        N = len(vocabulary)
        for vocab in vocabulary:
            res = {}
            for label in priors.keys():
                count = 1 # start at 1 for laplace smoothing
                for example in data:
                    if example[1] == label:
                        for word in example[0].split():
                            if vocab.lower() in word.lower():
                                count += 1
                N += 1
                res[label] = count / N
            cond_probs[vocab] = res
        self.priors = priors
        self.cond_probs = cond_probs

    def predict(self, document):
        '''
        1. Use same vectorizer in fit to transform document
        2. Use Bayes thm to calculate posterior probability of email being in each class P(class | email)
        3. Take class/label with highest probability
        '''
        doc_counts = self.vectorizer.transform(document)
        class_probs = defaultdict(float)
        for label in self.priors.keys():
            prob = self.priors[label]
            for (k,v) in self.vectorizer.vocabulary_.items(): # k - word, v - integer its mapped to
                p = self.cond_probs[k][label]
                x = doc_counts[(0,v)]
                prob *= p**x
            class_probs[label] = prob
        
        predicted_class = max(class_probs, key=class_probs.get)
        print(predicted_class)

def main():
    documents_with_labels = [('Free promotion for the best lottery giveaway ever! Vacation!', 1),
             ('Your meeting is at 2 pm. Do not be late please.', 0),
             ('Dear sir or maam, please accept the free promotion to win the best gift', 1),
             ('The meeting is running late at the company. Can you forward this?', 0)]

    NB = NaiveBayes()
    NB.fit(documents_with_labels)

    new_doc = ['Free giveaway! This is the best thing ever!']
    NB.predict(new_doc)


if __name__ == "__main__":
    main()