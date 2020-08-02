#pylint: disable-all
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class WordAccuracyTesting:

    def __init__(self):
        self.lowSampleSizeSentiments = []

    def predict(self, sentiments, sentiment, message, correctScore):
        if sentiment in sentiments:
            sentDict = sentiments[sentiment]
        elif sentiment not in sentiments:
            if sentiment not in self.lowSampleSizeSentiments:
                self.lowSampleSizeSentiments.append(sentiment)
            return [False, 0]
        words = word_tokenize(message)
        root_words = []
        for w in words:
            root_words.append(lemmatizer.lemmatize(w))
        c = Counter(sentDict)
        common = c.most_common(5)
        present = [False,False,False,False,False]
        lim = 0
        if len(common) >= len(root_words):
            lim = len(root_words)
        elif len(root_words) > len(common):
            lim = len(common)
        for i in range(0,lim):
            if root_words[i] == common[i][0]:
                present[i] == True
        weights = [0, 0, 0 ,0, 0]
        denom = 0
        for i in range(0,len(common)):
            denom += common[i][1]
        total = 0
        for i in range(0,len(common)):
            weights[i] = (common[i][1]) / denom
        for i in range(0,len(present)):
            if present[i] == True:
                total += weights[i]
        if total >= (weights[len(common) - 2]):
            answer = 1
        elif total >= (weights[len(common) - 1]):
            answer = 2
        elif total < (weights[len(common) - 1]):
            answer = 3
        if answer == correctScore:
            return [True, answer]
        else:
            return [False, answer]

    def runAccuracyTest(self, sentiments, sentiment, message, correctScore):
        noSentiment = False
        noScore = False
        if not sentiment in sentiments:
            noSentiment = True
        elif np.isnan(correctScore):
            noScore = True
        predict = self.predict(sentiments, sentiment, message, correctScore)
        result = [predict[0], predict[1], noSentiment, noScore]
        #print(result)
        self.updateSentimentWords(sentiment, sentiments, message)
        return result

    def updateSentimentWords(self, sentiment, sentiments, message):
        if sentiment in sentiments:
            initial_stopwords = set(stopwords.words('English'))
            stop_words = []
            for w in initial_stopwords:
                stop_words.append((lemmatizer.lemmatize(w)).lower())
            custom_sw = ( 'we', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\'s', 'a', 'in', 'patient', 'product', 'treatment', 'if', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'do', 'go365', 'pfizer')
            message_tokens = word_tokenize(message)
            for tok in message_tokens:
                root = lemmatizer.lemmatize(str(tok))
                root = root.lower()
                if not root in stop_words and not root in custom_sw:
                    if not root in sentiments[sentiment]:
                        (sentiments[sentiment])[root] = 1
                    else:
                        (sentiments[sentiment])[root] += 1
        elif sentiment not in sentiments:
            self.addSentiment(sentiment, sentiments, message)
    
    def addSentiment(self, sentiment, sentiments, message):
        if sentiment not in sentiments:
            sentiments[sentiment] = {}
            initial_stopwords = set(stopwords.words('English'))
            stop_words = []
            for w in initial_stopwords:
                stop_words.append((lemmatizer.lemmatize(w)).lower())
            custom_sw = ( 'We', '$', '%', '.', '*', ',', '!', ':', '?', ';', '(', ')', '\'s', 'A', 'In', 'patient', 'product', 'treatment', 'If', 'dont', 'I', 'n\'t', '-', 'help', 'bosulif', 'Do', 'go365', 'pfizer')
            message_tokens = word_tokenize(message)
            for tok in message_tokens:
                root = lemmatizer.lemmatize(str(tok))
                if not root in stop_words and not root in custom_sw:
                    if not root in sentiments[sentiment]:
                        (sentiments[sentiment])[root] = 1
                    else:
                        (sentiments[sentiment])[root] += 1
    
