#pylint: disable-all
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from collections import Counter

bigram_measures = BigramAssocMeasures()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class PhraseAccuracyTesting:

    def __init__(self):
        self.lowSampleSizeSentiments = []

    def predict(self, sentimentPhrases, sentiment, message, correctScore):
        if sentiment in sentimentPhrases:
            sentDict = sentimentPhrases[sentiment]
        elif sentiment not in sentimentPhrases:
            if sentiment not in self.lowSampleSizeSentiments:
                self.lowSampleSizeSentiments.append(sentiment)
            return [False,0]
        c = Counter(sentDict)
        phrases = c.most_common(5)
        #print(phrases)
        
        present = [False,False,False,False,False]
        lim = 0
        if len(phrases) >= len(present):
            lim = len(present)
        elif len(present) > len(phrases):
            lim = len(phrases)
        for i in range(0, lim):
            if phrases[i][0] in message:
                present[i] = True
        total = 0
        add = 5
        for pres in present:
            if(pres == True):
                total += add
            add-=1
        if total >= 5:
            answer = 1
        elif total >= 3:
            answer = 2
        elif total < 3:
            answer = 3
        if answer == correctScore:
            return [True, answer]
        else:
            return [False, answer]

    def runAccuracyTest(self, sentimentPhrases, sentiment, message, correctScore):
        noSentiment = False
        noScore = False
        if not sentiment in sentimentPhrases:
            noSentiment = True
        elif np.isnan(correctScore):
            noScore = True
        predict = self.predict(sentimentPhrases, sentiment, message, correctScore)
        result = [predict[0], predict[1], noSentiment, noScore]
        #print(result)
        self.updateSentimentPhrases(sentiment, sentimentPhrases, message)
        return result

    def updateSentimentPhrases(self, sentiment, sentimentPhrases, message):
        if sentiment in sentimentPhrases:
            finder = BigramCollocationFinder.from_words(message)
            phrases = finder.nbest(bigram_measures.pmi, 6)
            for p in phrases:
                ph = str(p[0]) + ' ' + str(p[1])
                if ph not in sentimentPhrases[sentiment]:
                    sentimentPhrases[sentiment][ph] = 1
                elif ph in sentimentPhrases[sentiment]:
                    sentimentPhrases[sentiment][ph] += 1
        elif sentiment not in sentimentPhrases:
            self.addSentiment(sentiment, sentimentPhrases, message)
            
    def addSentiment(self, sentiment, sentimentPhrases, message):
        if sentiment not in sentimentPhrases:
            sentimentPhrases[sentiment] = {}
            finder = BigramCollocationFinder.from_words(message)
            finder.apply_freq_filter(2)
            phrases = finder.nbest(bigram_measures.pmi, 6)
            phraseList = []
            for p in phrases:
                ph = str(p[0]) + ' ' + str(p[1])
                phraseList.append(ph)
            for ph in phraseList:
                if ph not in sentimentPhrases[sentiment]:
                    sentimentPhrases[sentiment][ph] = 1
                elif ph in sentimentPhrases[sentiment]:
                    sentimentPhrases[sentiment][ph] += 1
    
