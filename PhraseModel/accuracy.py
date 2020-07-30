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
        self.lowSampleSizeHeuristics = []

    def predict(self, heuristicPhrases, heuristic, message, correctScore):
        if heuristic in heuristicPhrases:
            heuDict = heuristicPhrases[heuristic]
        elif heuristic not in heuristicPhrases:
            if heuristic not in self.lowSampleSizeHeuristics:
                self.lowSampleSizeHeuristics.append(heuristic)
            return [False,0]
        c = Counter(heuDict)
        phrases = c.most_common(5)
        print(phrases)
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

    def runAccuracyTest(self, heuristicPhrases, heuristic, message, correctScore):
        noHeuristic = False
        noScore = False
        if not heuristic in heuristicPhrases:
            noHeuristic = True
        elif np.isnan(correctScore):
            noScore = True
        predict = self.predict(heuristicPhrases, heuristic, message, correctScore)
        result = [predict[0], predict[1], noHeuristic, noScore]
        #print(result)
        self.updateHeuristicPhrases(heuristic, heuristicPhrases, message)
        return result

    def updateHeuristicPhrases(self, heuristic, heuristicPhrases, message):
        if heuristic in heuristicPhrases:
            finder = BigramCollocationFinder.from_words(message)
            phrases = finder.nbest(bigram_measures.pmi, 6)
            for p in phrases:
                ph = str(p[0]) + ' ' + str(p[1])
                if ph not in heuristicPhrases[heuristic]:
                    heuristicPhrases[heuristic][ph] = 1
                elif ph in heuristicPhrases[heuristic]:
                    heuristicPhrases[heuristic][ph] += 1
        elif heuristic not in heuristicPhrases:
            self.addHeuristic(heuristic, heuristicPhrases, message)
            
    def addHeuristic(self, heuristic, heuristicPhrases, message):
        if heuristic not in heuristicPhrases:
            heuristicPhrases[heuristic] = {}
            finder = BigramCollocationFinder.from_words(message)
            finder.apply_freq_filter(2)
            phrases = finder.nbest(bigram_measures.pmi, 6)
            phraseList = []
            for p in phrases:
                ph = str(p[0]) + ' ' + str(p[1])
                phraseList.append(ph)
            for ph in phraseList:
                if ph not in heuristicPhrases[heuristic]:
                    heuristicPhrases[heuristic][ph] = 1
                elif ph in heuristicPhrases[heuristic]:
                    heuristicPhrases[heuristic][ph] += 1
    
